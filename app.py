import streamlit as st
import pandas as pd
from pathlib import Path
import inference

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto MHTD Forecaster",
    page_icon="₿",
    layout="wide",
)

# ── CUSTOM CRYPTO DARK THEME (CSS) ───────────────────────────
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0A0A0A;
        color: #E0E0E0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #2A2A2A;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F7931A !important; /* Bitcoin Orange */
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-shadow: 0px 0px 8px rgba(247, 147, 26, 0.4);
    }
    
    /* Cards & Metrics */
    div[data-testid="metric-container"] {
        background-color: #1A1A1A;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    div[data-testid="metric-container"] label {
        color: #888888 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #F7931A;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFA733;
        box-shadow: 0px 0px 15px rgba(247, 147, 26, 0.6);
        color: #000000;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #121212;
        border-radius: 8px;
        border: 1px solid #2A2A2A;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888888;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #F7931A !important;
        border-bottom-color: #F7931A !important;
    }
    
    /* Crypto Background Watermark */
    .crypto-bg {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 40vw;
        opacity: 0.02;
        z-index: -1;
        pointer-events: none;
        color: #FFFFFF;
    }
</style>
<div class="crypto-bg">₿</div>
""", unsafe_allow_html=True)

ART_DIR = Path(__file__).parent / "deployment_artifacts"

DISPLAY_NAMES = {
    "CatBoost": "MHTD-CatBoost (Proposed)",
    "Random Forest": "Random Forest",
    "Best": "Auto-Select (CatBoost)",
}

try:
    COINS = inference.load_artifacts(repo_id="sajia2217/crypto-forecaster-artifacts")
except Exception as e:
    st.error(f"Failed to load artifacts from `{ART_DIR}`. Did you run the training script first?\nError: {e}")
    st.stop()

# ── HEADER ──────────────────────────────────────────────────
st.title("₿ Crypto Multi-Horizon Forecaster")
st.markdown("##### *Institutional-grade 48-horizon forecasting using MHTD-CatBoost & Random Forest*")
st.divider()

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    coin = st.selectbox("Select Asset", COINS, format_func=lambda c: inference.COINS_META[c]["coin"])
    
    available = inference.list_models(coin)
    model_options = ["Best"] + available
    model = st.selectbox("Select Model", model_options, format_func=lambda m: DISPLAY_NAMES.get(m, m))
    
    horizons = inference.CONFIG["HORIZON_MINUTES"]
    h_min = st.select_slider(
        "Forecast Horizon",
        options=horizons,
        value=1440,
        format_func=lambda m: f"{m}m ({m//60:g}h)",
    )
    
    st.divider()
    # This defines the run_btn that was causing the error
    run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    st.caption(f"Trained models for {coin}: {', '.join(available)}")

# ── MAIN APPLICATION TABS ───────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction", 
    "📊 All 48 Horizons", 
    "📈 Past Evaluation", 
    "🔴 Live Market (Binance API)"
])

# Logic for Tabs 1-3 (Historical/Test Set predictions)
if not run_btn:
    with tab1:
        st.info("Configure your inputs in the sidebar and click **Generate Forecast**.")
    with tab2:
        st.info("Click **Generate Forecast** to view the 48-horizon grid.")
    with tab3:
        st.info("Click **Generate Forecast** to view past evaluation metrics.")
else:
    with st.spinner("Executing model inference..."):
        try:
            out, row = inference.predict_horizon(coin, h_min, model_name=model)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    # Tab 1: Single Prediction
    with tab1:
        st.markdown(f"#### {out['coin']} - {h_min} Minute Forecast")
        st.markdown(f"*Data Timestamp: {out['latest_time']}*")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Close", f"${out['latest_close']:,.4f}")
        c2.metric("Predicted Price", f"${row['Predicted Price']:,.4f}", delta=f"{row['Predicted Return (%)']:+.2f}%")
        
        dir_color = "normal" if row["Direction Call"] == "UP" else "inverse"
        c3.metric("Direction Signal", row["Direction Call"], delta=f"{row['Direction Probability (UP)']:.3f} prob", delta_color=dir_color)
        
        conf_status = "✅ YES" if row['Confident Setup'] else "⚠️ NO"
        c4.metric("High Conviction Gate", conf_status)
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Target Time:** `{row['Target Time']}`")
            st.markdown(f"**Model Used:** `{DISPLAY_NAMES.get(out['model'], out['model'])}`")
        with col_b:
            st.markdown(f"**Decision Threshold:** `{row['Threshold']:.4f}`")
            st.markdown(f"**Up Probability:** `{row['Direction Probability (UP)']:.4f}`")
            
        st.caption(f"📐 *Math: ${out['latest_close']:,.2f} × exp({row['Predicted Return (%)']/100:.4f}) = ${row['Predicted Price']:,.2f}*")

    # Tab 2: All Horizons
    with tab2:
        st.markdown(f"#### Full 48-Horizon Grid - {out['coin']}")
        st.dataframe(
            out["table"].style.format({
                "Latest Close": "{:,.4f}",
                "Predicted Price": "{:,.4f}",
                "Predicted Return (%)": "{:+.3f}",
                "Direction Probability (UP)": "{:.4f}",
                "Threshold": "{:.4f}",
            }).map(lambda x: "color: #00FF00;" if x == "UP" else "color: #FF0000;", subset=["Direction Call"]),
            use_container_width=True,
            height=600,
        )

    # Tab 3: Evaluation
    with tab3:
        st.markdown(f"#### Test-Set Evaluation - {DISPLAY_NAMES.get(out['model'], out['model'])}")
        ev = inference.load_eval_table(coin, out["model"])
        if ev is None:
            st.warning("Evaluation metrics CSV not found.")
        else:
            st.dataframe(ev.round(4), use_container_width=True, height=600)


# Tab 4: Live Binance API (Independent of the sidebar button)
with tab4:
    st.markdown(f"#### 🔴 Live Binance Forecast - {coin}")
    st.markdown("*This bypasses the saved historical data and fetches real-time 30m candles directly from Binance.*")
    
    live_btn = st.button("📡 Fetch Live Data & Predict", type="secondary")
    
    if live_btn:
        with st.spinner("Connecting to Binance API and running live inference..."):
            try:
                live_out, live_row = inference.predict_live(coin, h_min, model_name=model)
            except Exception as e:
                st.error(f"Live fetch failed: {e}")
                st.stop()
                
        st.success("Live data fetched successfully from Binance!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Close", f"${live_out['latest_close']:,.4f}")
        c2.metric("Live Predicted Price", f"${live_row['Predicted Price']:,.4f}", delta=f"{live_row['Predicted Return (%)']:+.2f}%")
        
        dir_color = "normal" if live_row["Direction Call"] == "UP" else "inverse"
        c3.metric("Direction Signal", live_row["Direction Call"], delta=f"{live_row['Direction Probability (UP)']:.3f} prob", delta_color=dir_color)
        
        c4.metric("High Conviction Gate", "✅ YES" if live_row['Confident Setup'] else "⚠️ NO")
        
        st.caption(f"📐 *Live Math: ${live_out['latest_close']:,.2f} × exp({live_row['Predicted Return (%)']/100:.4f}) = ${live_row['Predicted Price']:,.2f}*")
        st.markdown(f"**Live Data Timestamp:** `{live_out['latest_time']}`")
        
        st.divider()
        st.dataframe(live_out["table"].style.format({
            "Latest Close": "{:,.4f}", 
            "Predicted Price": "{:,.4f}", 
            "Predicted Return (%)": "{:+.3f}",
            "Direction Probability (UP)": "{:.4f}", 
            "Threshold": "{:.4f}",
        }).map(lambda x: "color: #00FF00;" if x == "UP" else "color: #FF0000;", subset=["Direction Call"]), 
        use_container_width=True, height=400)
    else:
        st.info("Click the button above to fetch today's live prices from Binance.")