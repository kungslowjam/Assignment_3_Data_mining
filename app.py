# ============================================================
# COMP7707 A3 - Streamlit Dashboard (Final v9 - Fixed Grid Layout)
# Member A - System Design & Implementation
# ============================================================

import os
import time
import psutil
import pandas as pd
import streamlit as st
from datetime import datetime

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
OUTPUT_FILE = "stream_output.csv"

st.set_page_config(page_title="🌦️ IoT Weather Anomaly Dashboard", layout="wide")

# Custom CSS (fix alignment)
st.markdown("""
    <style>
        /* Center header */
        h1 { text-align: center; }

        /* Make metric boxes equal width and aligned */
        div[data-testid="stMetric"] {
            background-color: rgba(32, 32, 32, 0.6);
            border-radius: 10px;
            padding: 10px 0;
            text-align: center;
            height: 100px;
            justify-content: center;
        }

        div[data-testid="stHorizontalBlock"] > div {
            flex: 1 1 0% !important;
            min-width: 180px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>🌦️ Real-time IoT Weather Anomaly Detection Dashboard</h1>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 1. SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")
refresh_sec = st.sidebar.slider("Auto-refresh interval (seconds)", 2, 20, 5)
max_rows = st.sidebar.slider("Rows to display in charts", 50, 1000, 300, step=50)
show_table = st.sidebar.checkbox("Show anomaly table", True)
show_all = st.sidebar.checkbox("Show all features", False)
auto_refresh = st.sidebar.checkbox("🔁 Auto refresh", True)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
@st.cache_data(ttl=5)
def load_data(path: str):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# ------------------------------------------------------------
# 3. UPDATE DASHBOARD
# ------------------------------------------------------------
def update_dashboard(df):
    if df is None or len(df) == 0:
        st.info("⚠️ Waiting for stream data...")
        return

    df_disp = df.tail(max_rows).copy()

    # --- Metrics calculation ---
    n_if = int(df_disp["IF_Flag"].sum()) if "IF_Flag" in df_disp.columns else 0
    n_oc = int(df_disp["OC_Flag"].sum()) if "OC_Flag" in df_disp.columns else 0
    n_gt = int(df_disp["GT_Label"].sum()) if "GT_Label" in df_disp.columns else 0

    # Combine anomalies to prevent >100%
    if "IF_Flag" in df_disp.columns and "OC_Flag" in df_disp.columns:
        union_anoms = sum((df_disp["IF_Flag"] == 1) | (df_disp["OC_Flag"] == 1))
    else:
        union_anoms = n_if + n_oc

    anomaly_ratio = round((union_anoms / len(df_disp)) * 100, 2)
    agree = (
        sum((df_disp["IF_Flag"] == 1) & (df_disp["OC_Flag"] == 1)) / len(df_disp)
        if "IF_Flag" in df_disp.columns and "OC_Flag" in df_disp.columns
        else 0
    ) * 100

    cpu = psutil.cpu_percent()
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e6
    file_size = os.path.getsize(OUTPUT_FILE) / 1024

    # --------------------------------------------------------
    # KPI ROW 1 (Fixed Alignment)
    # --------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Records", len(df))
    with col2:
        st.metric("🚨 IF Alerts", n_if)
    with col3:
        st.metric("⚠️ OCSVM Alerts", n_oc)
    with col4:
        st.metric("🧪 Synthetic Anomalies", n_gt)
    with col5:
        st.metric("💾 File Size (KB)", f"{file_size:.1f}")

    # --------------------------------------------------------
    # KPI ROW 2
    # --------------------------------------------------------
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("🤝 Model Agreement (%)", f"{agree:.2f}")
    with col7:
        st.metric("⚡ Anomaly Ratio (%)", f"{anomaly_ratio:.2f}")
    with col8:
        st.metric("🧠 CPU Usage (%)", cpu)

    st.sidebar.metric("🧮 Memory (MB)", f"{mem:.1f}")

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------
    all_features = [
        c for c in df.columns
        if c not in ["Index", "IF_Flag", "OC_Flag", "GT_Label", "Time"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if show_all:
        selected = all_features
    else:
        selected = st.sidebar.multiselect(
            "📈 Select features to visualize:",
            options=all_features,
            default=["airtemp", "relativehumidity", "solar"]
            if "solar" in all_features else all_features[:3],
        )

    st.markdown("### 📡 Live Streaming Data Monitor")
    if len(selected) > 0 and "Time" in df_disp.columns:
        st.line_chart(df_disp.set_index("Time")[selected], width='stretch')
    else:
        st.warning("⚠️ Please select at least one numeric feature to visualize.")

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------
    if show_table:
        st.markdown("### 🔍 Latest Detected Anomalies")
        if "IF_Flag" in df_disp.columns and "OC_Flag" in df_disp.columns:
            anomalies = df_disp[(df_disp["IF_Flag"] == 1) | (df_disp["OC_Flag"] == 1)]
            if len(anomalies) > 0:
                st.dataframe(anomalies.tail(15), width='stretch')
            else:
                st.info("No anomalies detected in this window.")
        else:
            st.warning("Missing anomaly flag columns.")

    st.caption(f"🕒 Last update: {datetime.now().strftime('%H:%M:%S')} | Refresh every {refresh_sec}s")

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
if not os.path.exists(OUTPUT_FILE):
    st.warning("⚠️ Stream file not found. Please run **prototype_final_fixed.py** first.")
else:
    df = load_data(OUTPUT_FILE)
    update_dashboard(df)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()
