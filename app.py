import os, time, urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. CONFIG
# ============================================================
DATA_URL = (
    "https://data.gov.au/data/dataset/"
    "southern-grampians-weather-sensor-data/resource/"
    "82a5e953-00dc-42d6-9c07-3066bf800be3/"
    "download/SGSC_Weather_Sensor_Data.csv"
)
LOCAL_PATH = "SGSC_Weather_Sensor_Data.csv"

st.set_page_config(page_title="🌦️ IoT Weather Anomaly Detection", layout="wide")
st.title("🌦️ Real-time IoT Weather Anomaly Detection Prototype")

# ============================================================
# 1. LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    """Download & clean dataset."""
    if not os.path.exists(LOCAL_PATH):
        st.info("⬇️ Downloading dataset from data.gov.au ...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
    df = pd.read_csv(LOCAL_PATH, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Parse datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S", errors="coerce")
    else:
        for c in df.columns:
            if "date" in c:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df.rename(columns={c: "time"}, inplace=True)

    df = df.dropna(subset=["time"]).sort_values("time")
    features = [
        f
        for f in ["airtemp", "relativehumidity", "windspeed", "solar", "vapourpressure"]
        if f in df.columns
    ]
    df = df.dropna(subset=features)
    return df, features


df, features = load_data()
st.success(f"✅ Loaded {len(df)} records | Features used: {features}")

# ============================================================
# 2. SIDEBAR SETTINGS
# ============================================================
st.sidebar.header("⚙️ Simulation Settings")
train_ratio = st.sidebar.slider("Training ratio", 0.5, 0.9, 0.7, 0.05)
delay_sec = st.sidebar.slider("Stream delay (seconds)", 0.0, 1.0, 0.15, 0.05)
contamination = st.sidebar.slider("IsolationForest contamination", 0.01, 0.2, 0.05, 0.01)
nu_val = st.sidebar.slider("OneClassSVM nu", 0.01, 0.2, 0.05, 0.01)
synthetic = st.sidebar.checkbox("Inject synthetic anomalies", value=True)
n_synth = st.sidebar.slider("Number of synthetic anomalies", 50, 400, 150, 10)

# ============================================================
# 3. FEATURE ENGINEERING & SYNTHETIC ANOMALIES
# ============================================================
def inject_anoms(df_in, features, n_points=150, strength=4.0):
    df_out = df_in.copy()
    df_out["gt_anomaly"] = 0
    idxs = np.random.choice(len(df_out), size=min(n_points, len(df_out)), replace=False)
    sigma = df_out[features].std().replace(0, 1e-6)
    for i in idxs:
        cols = np.random.choice(features, size=np.random.randint(1, len(features)), replace=False)
        for c in cols:
            direction = np.random.choice([+1, -1])
            df_out.at[i, c] += direction * strength * sigma[c]
        df_out.at[i, "gt_anomaly"] = 1
    return df_out

train_end = int(len(df) * train_ratio)
train_df = df.iloc[:train_end].copy()
stream_df = df.iloc[train_end:].copy().reset_index(drop=True)
if synthetic:
    stream_df = inject_anoms(stream_df, features, n_synth)
    st.info(f"🔬 Injected {stream_df['gt_anomaly'].sum()} synthetic anomalies.")

# ============================================================
# 4. TRAIN MODELS
# ============================================================
X_train = train_df[features].values
if_model = IsolationForest(
    contamination=contamination, random_state=42, n_estimators=200
)
if_model.fit(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
ocsvm = OneClassSVM(kernel="rbf", nu=nu_val, gamma="scale")
ocsvm.fit(X_train_scaled)

# ============================================================
# 5. STREAMING SIMULATION
# ============================================================
st.subheader("📡 Real-time Streaming Simulation")
col1, col2 = st.columns(2)
chart_if = col1.line_chart(pd.DataFrame({"IF_score": []}))
chart_oc = col2.line_chart(pd.DataFrame({"OCSVM_score": []}))
placeholder_table = st.empty()

start = st.button("▶️ Start Streaming")
if start:
    scores_if, scores_oc, alerts = [], [], []

    progress = st.progress(0, text="Streaming in progress...")
    for i, row in stream_df.iterrows():
        x = row[features].values.reshape(1, -1)
        s_if = if_model.decision_function(x)[0]
        l_if = 0 if if_model.predict(x)[0] == 1 else 1

        x_scaled = scaler.transform(x)
        s_oc = ocsvm.decision_function(x_scaled)[0]
        l_oc = 0 if ocsvm.predict(x_scaled)[0] == 1 else 1

        scores_if.append(s_if)
        scores_oc.append(s_oc)

        chart_if.add_rows(pd.DataFrame({"IF_score": [s_if]}))
        chart_oc.add_rows(pd.DataFrame({"OCSVM_score": [s_oc]}))

        if l_if == 1 or l_oc == 1:
            alerts.append(
                {
                    "Index": i,
                    "Time": row["time"],
                    "IF": l_if,
                    "OC": l_oc,
                    "airtemp": row["airtemp"],
                    "humidity": row["relativehumidity"],
                }
            )

        if i % 25 == 0:
            progress.progress(i / len(stream_df), text=f"Streaming record {i}")
            placeholder_table.dataframe(pd.DataFrame(alerts).tail(5))

        time.sleep(delay_sec)

    progress.empty()
    st.success("✅ Streaming Finished")

    # Alert summary
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        st.subheader("⚠️ Detected Anomalies (Latest 10)")
        st.dataframe(df_alerts.tail(10))
        st.download_button(
            "💾 Download Alerts CSV",
            df_alerts.to_csv(index=False).encode("utf-8"),
            file_name="alerts.csv",
        )

        n_if = df_alerts["IF"].sum()
        n_oc = df_alerts["OC"].sum()
        st.write(f"**IsolationForest Alerts:** {n_if} | **OneClassSVM Alerts:** {n_oc}")
    else:
        st.info("No anomalies detected.")
