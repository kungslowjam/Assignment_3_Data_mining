# ============================================================
# COMP7707 A3 - Streamlit Dashboard (Final – IF + AE)
# Member A - System Design & Implementation
# ============================================================

import os
import time
import psutil
import pandas as pd
import streamlit as st
from datetime import datetime
import altair as alt

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
OUTPUT_FILE = "stream_output.csv"

st.set_page_config(page_title="🌦️ IoT Weather Anomaly Dashboard", layout="wide")

# Custom CSS (fix alignment)
st.markdown("""
    <style>
        h1 { text-align: center; }

        div[data-testid="stMetric"] {
            background-color: rgba(32, 32, 32, 0.6);
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

    # ----- Anomaly type (for colouring) -----
    if "IF_Flag" in df_disp.columns and "AE_Flag" in df_disp.columns:
        df_disp["AnomalyType"] = "Normal"
        both_mask = (df_disp["IF_Flag"] == 1) & (df_disp["AE_Flag"] == 1)
        if_mask   = (df_disp["IF_Flag"] == 1) & (df_disp["AE_Flag"] == 0)
        ae_mask   = (df_disp["IF_Flag"] == 0) & (df_disp["AE_Flag"] == 1)

        df_disp.loc[both_mask, "AnomalyType"] = "Both (IF & AE)"
        df_disp.loc[if_mask,   "AnomalyType"] = "IF only"
        df_disp.loc[ae_mask,   "AnomalyType"] = "AE only"
    else:
        df_disp["AnomalyType"] = "Unknown"

    # ----- KPIs -----
    n_if = int(df_disp["IF_Flag"].sum()) if "IF_Flag" in df_disp.columns else 0
    n_ae = int(df_disp["AE_Flag"].sum()) if "AE_Flag" in df_disp.columns else 0
    n_gt = int(df_disp["GT_Label"].sum()) if "GT_Label" in df_disp.columns else 0

    if "IF_Flag" in df_disp.columns and "AE_Flag" in df_disp.columns:
        union_anoms = ((df_disp["IF_Flag"] == 1) | (df_disp["AE_Flag"] == 1)).sum()
        agree = (
            ((df_disp["IF_Flag"] == 1) & (df_disp["AE_Flag"] == 1)).sum()
            / len(df_disp)
        ) * 100
    else:
        union_anoms = n_if + n_ae
        agree = 0.0

    anomaly_ratio = round((union_anoms / len(df_disp)) * 100, 2)

    cpu = psutil.cpu_percent()
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e6

    # Top KPI row (4 metrics)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", len(df))
    with col2:
        st.metric("🚨 IF Alerts", n_if)
    with col3:
        st.metric("🤖 AE Alerts", n_ae)
    with col4:
        st.metric("🧪 Synthetic Anomalies", n_gt)

    # Second KPI row
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("🤝 Model Agreement (%)", f"{agree:.2f}")
    with col7:
        st.metric("⚡ Anomaly Ratio (%)", f"{anomaly_ratio:.2f}")
    with col8:
        st.metric("🧠 CPU Usage (%)", cpu)

    st.sidebar.metric("🧮 Memory (MB)", f"{mem:.1f}")

    # --------------------------------------------------------
    # CHARTS – per-feature with anomaly points
    # --------------------------------------------------------
    all_features = [
        c for c in df.columns
        if c not in ["Index", "IF_Flag", "AE_Flag", "GT_Label", "Time", "AnomalyType"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if show_all:
        selected = all_features
    else:
        default_feats = ["airtemp", "relativehumidity", "solar"]
        default = [f for f in default_feats if f in all_features]
        if not default and len(all_features) > 0:
            default = all_features[:3]

        selected = st.sidebar.multiselect(
            "📈 Select features to visualize:",
            options=all_features,
            default=default,
        )

    st.markdown("### 📡 Live Streaming Data Monitor")

    if len(selected) > 0 and "Time" in df_disp.columns:
        anomaly_color_scale = alt.Scale(
            domain=["IF only", "AE only", "Both (IF & AE)", "Normal"],
            range=["#FFA500", "#1E88E5", "#FF0000", "#AAAAAA"],
        )

        for feat in selected:
            st.markdown(f"#### {feat}")

            base = alt.Chart(df_disp).mark_line().encode(
                x=alt.X("Time:T", title="Time"),
                y=alt.Y(f"{feat}:Q", title="Sensor Value"),
                tooltip=[
                    alt.Tooltip("Time:T"),
                    alt.Tooltip(f"{feat}:Q", format=".2f"),
                ],
            )

            df_anom = df_disp[
                ((df_disp["IF_Flag"] == 1) | (df_disp["AE_Flag"] == 1))
                & df_disp[feat].notna()
            ]

            if not df_anom.empty:
                points = alt.Chart(df_anom).mark_circle(size=70).encode(
                    x="Time:T",
                    y=f"{feat}:Q",
                    color=alt.Color(
                        "AnomalyType:N",
                        title="Anomaly Type",
                        scale=anomaly_color_scale,
                    ),
                    tooltip=[
                        alt.Tooltip("Time:T"),
                        alt.Tooltip(f"{feat}:Q", format=".2f"),
                        alt.Tooltip("AnomalyType:N", title="Type"),
                        alt.Tooltip("IF_Flag:Q", title="IF Flag"),
                        alt.Tooltip("AE_Flag:Q", title="AE Flag"),
                        alt.Tooltip("GT_Label:Q", title="GT Label"),
                    ],
                )
                chart = (base + points).properties(height=230).interactive()
            else:
                chart = base.properties(height=230).interactive()

            st.altair_chart(chart, width="stretch")

    else:
        st.warning("⚠️ Please select at least one numeric feature to visualize.")

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------
    if show_table:
        st.markdown("### 🔍 Latest Detected Anomalies")
        if "IF_Flag" in df_disp.columns and "AE_Flag" in df_disp.columns:
            anomalies = df_disp[(df_disp["IF_Flag"] == 1) | (df_disp["AE_Flag"] == 1)]
            if len(anomalies) > 0:
                st.dataframe(anomalies.tail(15), width="stretch")
            else:
                st.info("No anomalies detected in this window.")
        else:
            st.warning("Missing anomaly flag columns.")

    st.caption(
        f"🕒 Last update: {datetime.now().strftime('%H:%M:%S')} | "
        f"Refresh every {refresh_sec}s"
    )

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
if not os.path.exists(OUTPUT_FILE):
    st.warning("⚠️ Stream file not found. Please run **prototype.py** first.")
else:
    df = load_data(OUTPUT_FILE)
    update_dashboard(df)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()
