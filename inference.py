import json
import re
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from huggingface_hub import snapshot_download # <-- NEW IMPORT

warnings.filterwarnings("ignore")

CONFIG = None
N_HORIZONS = None
LOOKBACK = None
FEATURE_COLS = None
RETURN_LAGS = None
MOMENTUM_WINDOWS = None
EXPECTED_INTERVAL = None
BTC_SIGNALS = None
DEPLOY_ROOT = None
COINS_META = {}

def safe_keras_name(text):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text))

def load_artifacts(repo_id="sajia2217/crypto-forecaster-artifacts"): # <-- CHANGE THIS
    global CONFIG, N_HORIZONS, LOOKBACK, FEATURE_COLS, RETURN_LAGS
    global MOMENTUM_WINDOWS, EXPECTED_INTERVAL, BTC_SIGNALS, DEPLOY_ROOT, COINS_META

    # 1. Download the entire folder from Hugging Face
    print(f"Downloading artifacts from Hugging Face: {repo_id}...")
    local_dir = snapshot_download(repo_id=repo_id, repo_type="model")
    
    # 2. Set the deployment root to the downloaded directory
    DEPLOY_ROOT = Path(local_dir)

    with open(DEPLOY_ROOT / "config.json") as f:
        CONFIG = json.load(f)

    N_HORIZONS = CONFIG["N_HORIZONS"]
    LOOKBACK = CONFIG["LOOKBACK"]
    FEATURE_COLS = CONFIG["FEATURE_COLS"]
    RETURN_LAGS = CONFIG["RETURN_LAGS"]
    MOMENTUM_WINDOWS = CONFIG["MOMENTUM_WINDOWS"]
    EXPECTED_INTERVAL = pd.Timedelta(minutes=CONFIG["EXPECTED_INTERVAL_MINUTES"])

    BTC_SIGNALS = pd.read_parquet(DEPLOY_ROOT / "btc_market_signals.parquet")
    BTC_SIGNALS.index = pd.to_datetime(BTC_SIGNALS.index)

    COINS_META = {}
    # The snapshot_download might place files in a subfolder or directly. 
    # We iterate through to find the metadata.json files.
    for coin_dir in sorted(DEPLOY_ROOT.iterdir()):
        if not coin_dir.is_dir():
            continue
        meta_path = coin_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        COINS_META[safe_keras_name(meta["coin"])] = meta
        
    return list(COINS_META.keys())


# ============================================================
# FEATURE ENGINEERING (Copied exactly from training script)
# ============================================================
def _seg_group(frame):               return frame.groupby("_segment", group_keys=False)
def _seg_transform(frame, col, fn):  return _seg_group(frame)[col].transform(fn)

def build_feature_frame(raw_coin_df):
    frame = raw_coin_df.copy()
    frame["Date & Time"] = pd.to_datetime(frame["Date & Time"])
    frame = frame.sort_values("Date & Time").reset_index(drop=True).copy()
    frame["_segment"] = frame["Date & Time"].diff().ne(EXPECTED_INTERVAL).cumsum()
    eps = 1e-9

    grouped_close      = _seg_group(frame)["Close"]
    frame["Return"]     = grouped_close.pct_change()
    frame["Log_Return"] = np.log(frame["Close"] / grouped_close.shift(1))

    frame["Price_Change"]       = frame["Close"] - frame["Open"]
    frame["Price_Change_Pct"]   = frame["Price_Change"] / (frame["Open"] + eps) * 100
    frame["High_Low_Range"]     = frame["High"] - frame["Low"]
    frame["High_Low_Range_Pct"] = frame["High_Low_Range"] / (frame["Open"] + eps) * 100
    frame["Close_Open_Ratio"]   = frame["Close"] / (frame["Open"] + eps)

    for w in (12, 24):
        frame[f"Rolling_Mean_{w}"] = _seg_transform(frame, "Close", lambda s, w=w: s.rolling(w).mean())
        frame[f"Rolling_Std_{w}"]  = _seg_transform(frame, "Close", lambda s, w=w: s.rolling(w).std())
    for span in (10, 20, 50, 100, 200):
        frame[f"EMA_{span}"] = _seg_transform(
            frame, "Close", lambda s, span=span: s.ewm(span=span, adjust=False).mean())

    frame["Trend_Slope_10"]        = _seg_transform(frame, "EMA_10",  lambda s: s.pct_change(3).fillna(0))
    frame["Trend_Slope_20"]        = _seg_transform(frame, "EMA_20",  lambda s: s.pct_change(6).fillna(0))
    frame["Trend_Slope_50"]        = _seg_transform(frame, "EMA_50",  lambda s: s.pct_change(12).fillna(0))
    frame["Macro_Trend_Slope_100"] = _seg_transform(frame, "EMA_100", lambda s: s.pct_change(24).fillna(0))
    frame["Macro_Trend_Slope_200"] = _seg_transform(frame, "EMA_200", lambda s: s.pct_change(48).fillna(0))
    frame["EMA_Spread_10_50"]      = (frame["EMA_10"] - frame["EMA_50"]) / (frame["EMA_50"] + eps)
    frame["EMA_Spread_20_50"]      = (frame["EMA_20"] - frame["EMA_50"]) / (frame["EMA_50"] + eps)
    frame["Macro_Spread_50_200"]   = (frame["EMA_50"] - frame["EMA_200"]) / (frame["EMA_200"] + eps)
    frame["Macro_Spread_10_100"]   = (frame["EMA_10"] - frame["EMA_100"]) / (frame["EMA_100"] + eps)

    def rsi(close, w):
        d = close.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        rs = g.rolling(w).mean() / (l.rolling(w).mean() + eps)
        return 100 - 100 / (1 + rs)
    frame["RSI_14"]       = _seg_transform(frame, "Close", lambda s: rsi(s, 14))
    frame["Macro_RSI_48"] = _seg_transform(frame, "Close", lambda s: rsi(s, 48))

    ema12 = _seg_transform(frame, "Close", lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = _seg_transform(frame, "Close", lambda s: s.ewm(span=26, adjust=False).mean())
    frame["MACD"]        = ema12 - ema26
    frame["MACD_Signal"] = _seg_transform(frame, "MACD", lambda s: s.ewm(span=9, adjust=False).mean())
    frame["BB_Middle"]   = _seg_transform(frame, "Close", lambda s: s.rolling(20).mean())
    bb_std               = _seg_transform(frame, "Close", lambda s: s.rolling(20).std())
    frame["BB_Upper"]    = frame["BB_Middle"] + 2 * bb_std
    frame["BB_Lower"]    = frame["BB_Middle"] - 2 * bb_std

    for lag in RETURN_LAGS:
        frame[f"Log_Return_Lag_{lag}"] = _seg_group(frame)["Log_Return"].shift(lag)
    for w in MOMENTUM_WINDOWS:
        frame[f"Momentum_{w}"] = _seg_transform(frame, "Log_Return", lambda s, w=w: s.rolling(w).sum())

    frame["Volume_Change"]   = _seg_group(frame)["Volume"].pct_change()
    frame["Trades_Change"]   = _seg_group(frame)["Trades"].pct_change()
    frame["Volume_Ratio_12"] = frame["Volume"] / (_seg_transform(frame, "Volume", lambda s: s.rolling(12).mean()) + eps)
    frame["Trades_Ratio_12"] = frame["Trades"] / (_seg_transform(frame, "Trades", lambda s: s.rolling(12).mean()) + eps)
    v6  = _seg_transform(frame, "Log_Return", lambda s: s.rolling(6).std())
    v24 = _seg_transform(frame, "Log_Return", lambda s: s.rolling(24).std())
    v96 = _seg_transform(frame, "Log_Return", lambda s: s.rolling(96).std())
    frame["Volatility_Ratio_6_24"]  = v6 / (v24 + eps)
    frame["Volatility_Ratio_24_96"] = v24 / (v96 + eps)

    frame["RSI_Change"]      = _seg_group(frame)["RSI_14"].diff()
    frame["MACD_Histogram"]  = frame["MACD"] - frame["MACD_Signal"]
    frame["Close_EMA10_Gap"] = (frame["Close"] - frame["EMA_10"]) / (frame["EMA_10"] + eps)
    frame["BB_Position"]     = ((frame["Close"] - frame["BB_Lower"]) /
                                (frame["BB_Upper"] - frame["BB_Lower"] + eps))

    prev_close = _seg_group(frame)["Close"].shift(1)
    tr = pd.concat([frame["High"] - frame["Low"],
                    (frame["High"] - prev_close).abs(),
                    (frame["Low"] - prev_close).abs()], axis=1).max(axis=1)
    frame["ATR_14"] = _seg_transform(frame, "Close", lambda s: tr.rolling(14).mean()) / (frame["Close"] + eps)
    dmove = (frame["Close"] - prev_close).abs()
    frame["ADX_Proxy"] = _seg_transform(frame, "Close", lambda s: dmove.rolling(14).mean()) / (
        frame["ATR_14"] * frame["Close"] + eps)

    obv_dir = np.sign(frame["Return"].fillna(0))
    obv_raw = (obv_dir * frame["Volume"]).cumsum()
    frame["OBV_Change"]         = _seg_transform(frame, "Volume", lambda s: obv_raw.pct_change(6).fillna(0))
    frame["Macro_OBV_Slope_48"] = _seg_transform(frame, "Volume", lambda s: obv_raw.pct_change(48).fillna(0))
    frame["Volume_Price_Trend"] = (frame["Return"].fillna(0) * frame["Volume"]) / (
        _seg_transform(frame, "Volume", lambda s: s.rolling(12).mean()) + eps)

    frame["Macro_Return_12h"] = _seg_transform(frame, "Close", lambda s: s.pct_change(24).fillna(0))
    frame["Macro_Return_24h"] = _seg_transform(frame, "Close", lambda s: s.pct_change(48).fillna(0))
    frame["Macro_Up_12h"]     = (frame["Macro_Return_12h"] > 0).astype(float)
    frame["Macro_Up_24h"]     = (frame["Macro_Return_24h"] > 0).astype(float)
    frame["Macro_Return_3d"]  = _seg_transform(frame, "Close", lambda s: s.pct_change(144).fillna(0))
    frame["Macro_Return_7d"]  = _seg_transform(frame, "Close", lambda s: s.pct_change(336).fillna(0))

    merged = frame[["Date & Time"]].merge(BTC_SIGNALS.reset_index(), on="Date & Time", how="left")
    btc_ret = merged["BTC_Log_Return"].fillna(0.0)
    btc_vol = merged["BTC_Vol_12"].fillna(0.0)
    btc_m12 = merged["BTC_Macro_Return_12h"].fillna(0.0)
    btc_m24 = merged["BTC_Macro_Return_24h"].fillna(0.0)
    frame["BTC_Lead_Lag_1"]           = btc_ret.shift(1).fillna(0.0)
    frame["BTC_Lead_Lag_2"]           = btc_ret.shift(2).fillna(0.0)
    frame["BTC_Lead_Lag_3"]           = btc_ret.shift(3).fillna(0.0)
    frame["BTC_Lead_Vol_12"]          = btc_vol.shift(1).fillna(0.0)
    frame["BTC_Macro_Return_12h"]     = btc_m12.shift(1).fillna(0.0)
    frame["BTC_Macro_Return_24h"]     = btc_m24.shift(1).fillna(0.0)
    frame["BTC_Relative_Strength_48"] = frame["Momentum_48"] - frame["BTC_Macro_Return_24h"]

    hour = frame["Date & Time"].dt.hour + frame["Date & Time"].dt.minute / 60
    day  = frame["Date & Time"].dt.dayofweek
    frame["Hour_Sin"] = np.sin(2 * np.pi * hour / 24)
    frame["Hour_Cos"] = np.cos(2 * np.pi * hour / 24)
    frame["Day_Sin"]  = np.sin(2 * np.pi * day / 7)
    frame["Day_Cos"]  = np.cos(2 * np.pi * day / 7)

    return (frame.replace([np.inf, -np.inf], np.nan)
                 .dropna(subset=FEATURE_COLS)
                 .reset_index(drop=True))


def compact_window_features(x_seq):
    return np.concatenate([
        x_seq[:, -1, :],
        x_seq.mean(axis=1),
        x_seq.std(axis=1),
        x_seq[:, -1, :] - x_seq[:, 0, :],
    ], axis=1).astype(np.float32)


def apply_temporal_decay_calibration(pred, shrink):
    p = np.asarray(pred, dtype=np.float32)
    s = np.asarray(shrink, dtype=np.float32)
    return p * s if p.ndim == 1 else p * s[None, :]

def extract_catboost_probability(clf, X):
    raw = np.asarray(clf.predict_proba(X))
    n, H = len(X), N_HORIZONS
    
    # Handle single-sample prediction (where CatBoost squeezes dimensions)
    if n == 1:
        if raw.ndim == 1 and raw.shape == (H,):
            return raw.reshape(1, H).astype(np.float32)
        if raw.ndim == 2 and raw.shape == (H, 2):
            return raw[:, 1].reshape(1, H).astype(np.float32)
        if raw.ndim == 2 and raw.shape == (1, H):
            return raw.astype(np.float32)
            
    # Handle batch prediction
    raw = np.squeeze(raw)
    if raw.shape == (n, H): return raw.astype(np.float32)
    if raw.shape == (H, n): return raw.T.astype(np.float32)
    if raw.ndim == 3 and raw.shape[-1] == 2:
        r = raw[..., 1]
        return (r.T if r.shape == (H, n) else r).astype(np.float32)
        
    # Fallback for 1D arrays
    if raw.ndim == 1 and len(raw) == H:
        return raw.reshape(1, H).astype(np.float32)
        
    raise ValueError(f"Unexpected CatBoost proba shape: {raw.shape}")
def rf_positive_probability(clf, X):
    proba_list = clf.predict_proba(X)
    out = np.zeros((len(X), N_HORIZONS), dtype=np.float32)
    for i, p in enumerate(proba_list):
        classes = np.asarray(clf.classes_[i])
        pos = np.flatnonzero(classes == 1)
        if len(pos):
            out[:, i] = p[:, pos[0]]
        elif len(classes) == 1 and classes[0] == 1:
            out[:, i] = 1.0
    return out

def _prepare_latest_inputs(coin_safe):
    meta = COINS_META[coin_safe]
    coin = meta["coin"]
    coin_dir = DEPLOY_ROOT / coin_safe

    raw = pd.read_parquet(DEPLOY_ROOT / "raw_data" / f"{coin}.parquet")
    raw["Date & Time"] = pd.to_datetime(raw["Date & Time"])

    frame = build_feature_frame(raw)
    if len(frame) < LOOKBACK:
        raise ValueError(f"Not enough data for {coin}.")

    seg_last = frame["_segment"].iloc[-1]
    tail = frame.loc[frame["_segment"] == seg_last].tail(LOOKBACK)
    
    x_scaler = joblib.load(coin_dir / "x_scaler.joblib")
    x = tail[FEATURE_COLS].to_numpy(dtype=np.float32)
    x_scaled = x_scaler.transform(x).astype(np.float32)[None, :, :]
    tree = compact_window_features(x_scaled)

    return {
        "seq": x_scaled, "tree": tree,
        "close": float(tail["Close"].iloc[-1]),
        "time": pd.Timestamp(tail["Date & Time"].iloc[-1]),
    }

def list_models(coin):
    meta = COINS_META.get(safe_keras_name(coin))
    if meta is None: return []
    return list(meta["available_models"].keys())

def predict_latest(coin, model_name="Best"):
    coin_safe = safe_keras_name(coin)
    meta = COINS_META[coin_safe]
    coin_dir = DEPLOY_ROOT / coin_safe

    if model_name == "Best":
        model_name = "CatBoost" if "CatBoost" in meta["available_models"] else list(meta["available_models"].keys())[0]

    latest = _prepare_latest_inputs(coin_safe)
    y_scaler = joblib.load(coin_dir / "y_scaler.joblib")
    kind = meta["available_models"][model_name]["kind"]

    if kind == "random_forest":
        bundle = joblib.load(coin_dir / "models" / "Random_Forest_multihorizon.joblib")
        pred_scaled = bundle["regressor"].predict(latest["tree"])[0]
        proba = rf_positive_probability(bundle["classifier"], latest["tree"])[0]
        return_raw = y_scaler.inverse_transform(pred_scaled[None, :])[0]
        thresholds = np.asarray(meta["direction_thresholds"].get("Random Forest", [0.5]*N_HORIZONS))

    elif kind == "catboost":
        bundle = joblib.load(coin_dir / "models" / "CatBoost_models.joblib")
        pred_scaled = bundle["regressor"].predict(latest["tree"])[0]
        proba = extract_catboost_probability(bundle["classifier"], latest["tree"])[0]
        calib = bundle.get("calibration_info")
        if calib and "shrinkage_vector" in calib:
            pred_scaled = apply_temporal_decay_calibration(pred_scaled, calib["shrinkage_vector"])
        return_raw = y_scaler.inverse_transform(pred_scaled[None, :])[0]
        thresholds = np.asarray(meta["direction_thresholds"].get("CatBoost", [0.5]*N_HORIZONS))

    close = latest["close"]
    prices = close * np.exp(return_raw)
    times = [latest["time"] + EXPECTED_INTERVAL * int(s) for s in CONFIG["HORIZON_STEPS"]]
    calls = np.where(proba >= thresholds, "UP", "DOWN")
    confident = np.abs(proba - thresholds) >= 0.10

    table = pd.DataFrame({
        "Target Time": times,
        "Horizon Minutes": CONFIG["HORIZON_MINUTES"],
        "Latest Close": close,
        "Predicted Price": prices,
        "Predicted Return (%)": return_raw * 100,
        "Direction Probability (UP)": proba,
        "Threshold": thresholds,
        "Direction Call": calls,
        "Confident Setup": confident,
    })

    return {
        "coin": coin, "model": model_name, "latest_time": latest["time"],
        "latest_close": close, "table": table,
    }

def predict_horizon(coin, minutes, model_name="Best"):
    minutes = int(minutes)
    idx = CONFIG["HORIZON_MINUTES"].index(minutes)
    out = predict_latest(coin, model_name)
    return out, out["table"].iloc[idx]

def load_eval_table(coin, model_name):
    coin_dir = DEPLOY_ROOT / safe_keras_name(coin)
    csv = coin_dir / "eval" / f"{safe_keras_name(model_name)}.csv"
    if not csv.exists(): return None
    return pd.read_csv(csv)




##Live market



def fetch_live_binance_data(coin):
    """Fetches the latest 600 klines from Binance public API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={coin}&interval=30m&limit=600"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Binance API error: {response.text}")
    
    raw_data = response.json()
    df = pd.DataFrame(raw_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", 
        "Close Time", "Quote Asset Volume", "Number of Trades", 
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ])
    
    # Convert to the format our pipeline expects
    df["Date & Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Turnover"] = df["Quote Asset Volume"]
    df["Trades"] = df["Number of Trades"]
    
    # Keep only required columns and convert to numeric
    cols = ["Date & Time", "Open", "High", "Low", "Close", "Volume", "Turnover", "Trades"]
    df = df[cols].copy()
    for c in ["Open", "High", "Low", "Close", "Volume", "Turnover", "Trades"]:
        df[c] = pd.to_numeric(df[c])
        
    return df

def predict_live(coin, minutes, model_name="Best"):
    """Runs the exact same inference pipeline but on live Binance data."""
    coin_safe = safe_keras_name(coin)
    meta = COINS_META[coin_safe]
    coin_dir = DEPLOY_ROOT / coin_safe

    if model_name == "Best":
        model_name = "CatBoost" if "CatBoost" in meta["available_models"] else list(meta["available_models"].keys())[0]

    # 1. Fetch live data
    raw_live = fetch_live_binance_data(coin)
    
    # 2. Build features
    frame = build_feature_frame(raw_live)
    if len(frame) < LOOKBACK:
        raise ValueError(f"Not enough live data for {coin}.")

    seg_last = frame["_segment"].iloc[-1]
    tail = frame.loc[frame["_segment"] == seg_last].tail(LOOKBACK)
    
    x_scaler = joblib.load(coin_dir / "x_scaler.joblib")
    x = tail[FEATURE_COLS].to_numpy(dtype=np.float32)
    x_scaled = x_scaler.transform(x).astype(np.float32)[None, :, :]
    tree = compact_window_features(x_scaled)

    latest = {
        "seq": x_scaled, "tree": tree,
        "close": float(tail["Close"].iloc[-1]),
        "time": pd.Timestamp(tail["Date & Time"].iloc[-1]),
    }

    # 3. Predict
    y_scaler = joblib.load(coin_dir / "y_scaler.joblib")
    kind = meta["available_models"][model_name]["kind"]

    if kind == "random_forest":
        bundle = joblib.load(coin_dir / "models" / "Random_Forest_multihorizon.joblib")
        pred_scaled = bundle["regressor"].predict(latest["tree"])[0]
        proba = rf_positive_probability(bundle["classifier"], latest["tree"])[0]
        return_raw = y_scaler.inverse_transform(pred_scaled[None, :])[0]
        thresholds = np.asarray(meta["direction_thresholds"].get("Random Forest", [0.5]*N_HORIZONS))
    elif kind == "catboost":
        bundle = joblib.load(coin_dir / "models" / "CatBoost_models.joblib")
        pred_scaled = bundle["regressor"].predict(latest["tree"])[0]
        proba = extract_catboost_probability(bundle["classifier"], latest["tree"])[0]
        calib = bundle.get("calibration_info")
        if calib and "shrinkage_vector" in calib:
            pred_scaled = apply_temporal_decay_calibration(pred_scaled, calib["shrinkage_vector"])
        return_raw = y_scaler.inverse_transform(pred_scaled[None, :])[0]
        thresholds = np.asarray(meta["direction_thresholds"].get("CatBoost", [0.5]*N_HORIZONS))

    close = latest["close"]
    prices = close * np.exp(return_raw)
    times = [latest["time"] + EXPECTED_INTERVAL * int(s) for s in CONFIG["HORIZON_STEPS"]]
    calls = np.where(proba >= thresholds, "UP", "DOWN")
    confident = np.abs(proba - thresholds) >= 0.10

    table = pd.DataFrame({
        "Target Time": times,
        "Horizon Minutes": CONFIG["HORIZON_MINUTES"],
        "Latest Close": close,
        "Predicted Price": prices,
        "Predicted Return (%)": return_raw * 100,
        "Direction Probability (UP)": proba,
        "Threshold": thresholds,
        "Direction Call": calls,
        "Confident Setup": confident,
    })

    out = {
        "coin": coin, "model": model_name, "latest_time": latest["time"],
        "latest_close": close, "table": table,
    }
    
    # Find the row for the requested horizon and return BOTH the table and the specific row
    idx = CONFIG["HORIZON_MINUTES"].index(int(minutes))
    return out, out["table"].iloc[idx]