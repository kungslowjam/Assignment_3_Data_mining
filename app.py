# ============================================================
# COMP7707 A3 - IoT Weather Anomaly Detection Dashboard
# Member A - System Design & Implementation
# ============================================================

import streamlit as st
import pandas as pd
import time
import os

# ------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------
DATA_FILE = "stream_output.csv"

st.set_page_config(page_title="🌦️ IoT Streaming Dashboard", layout="wide")
st.title("🌦️ Real-time IoT Weather Anomaly Detection Dashboard")

# ------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh interval (seconds)", 0.5, 5.0, 1.0, 0.5)
max_points = st.sidebar.slider("Max points displayed", 50, 1000, 200, 50)

# ------------------------------------------------------------
# 2. UI PLACEHOLDERS
# ------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)
chart_temp = chart_col1.empty()
chart_hum = chart_col2.empty()
table_placeholder = st.empty()
status_placeholder = st.sidebar.empty()

# ------------------------------------------------------------
# 3. REAL-TIME LOOP
# ------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.warning("⚠️ Waiting for stream_output.csv ... Please run prototype_local_stream.py first.")
else:
    st.success("✅ Connected to real-time data stream.")
    st.info("📡 Streaming live sensor data... Dashboard updates automatically!")

    last_size = 0
    while True:
        try:
            df = pd.read_csv(DATA_FILE)

            if df.empty:
                status_placeholder.warning("No data yet... waiting for stream...")
                time.sleep(refresh_rate)
                continue

            # Show only latest part of the stream
            df_tail = df.tail(max_points)

            # Plot Air Temperature
            if "AirTemp" in df.columns:
                chart_temp.line_chart(df_tail[["AirTemp"]], height=300)

            # Plot Humidity
            if "Humidity" in df.columns:
                chart_hum.line_chart(df_tail[["Humidity"]], height=300)

            # Display latest anomalies
            df_alerts = df_tail[(df_tail["IF_Flag"] == 1) | (df_tail["OC_Flag"] == 1)]
            table_placeholder.subheader("⚠️ Detected Anomalies (latest)")
            if not df_alerts.empty:
                table_placeholder.dataframe(df_alerts.tail(10))
            else:
                table_placeholder.info("No anomalies detected in recent window.")

            # Show stats
            total_rows = len(df)
            anomalies = (df["IF_Flag"] | df["OC_Flag"]).sum()
            status_placeholder.success(
                f"Records: {total_rows:,} | Anomalies detected: {anomalies}"
            )

            time.sleep(refresh_rate)

        except pd.errors.EmptyDataError:
            time.sleep(refresh_rate)
        except FileNotFoundError:
            st.warning("⚠️ File not found. Make sure prototype_local_stream.py is running.")
            time.sleep(refresh_rate)
        except KeyboardInterrupt:
            st.stop()
