# ============================================================
# COMP7707 A3 - Real-time IoT Analytics System Prototype
# Member A - System Design & Implementation
# ============================================================

import os
import time
import urllib.request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------
DATA_URL = (
    "https://data.gov.au/data/dataset/"
    "southern-grampians-weather-sensor-data/resource/"
    "82a5e953-00dc-42d6-9c07-3066bf800be3/"
    "download/SGSC_Weather_Sensor_Data.csv"
)
LOCAL_PATH = "SGSC_Weather_Sensor_Data.csv"
OUTPUT_FILE = "stream_output.csv"

TRAIN_RATIO = 0.7
CONTAMINATION = 0.05
NU_VAL = 0.05
STREAM_DELAY = 0.1   # seconds per record
SAMPLE_SIZE = 10000  # limit for faster testing
SYNTHETIC = True
SYNTH_POINTS = 150


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
def load_data():
    """Download dataset if needed and correctly parse timestamp from numeric."""
    if not os.path.exists(LOCAL_PATH):
        print("⬇️ Downloading dataset ...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
        print("✅ Download complete!")

    df = pd.read_csv(LOCAL_PATH, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # --- detect timestamp column automatically ---
    time_col = None
    for c in df.columns:
        if any(k in c for k in ["time", "date", "timestamp", "recorded"]):
            time_col = c
            break
    if not time_col:
        raise ValueError("❌ No time/date column found!")

    df.rename(columns={time_col: "time"}, inplace=True)

    # --- convert numeric timestamp (e.g. 2.01808E+13) to datetime ---
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    # milliseconds since epoch → datetime
    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    # --- detect main numeric features ---
    features = [
        f for f in ["airtemp", "relativehumidity", "windspeed", "solar", "vapourpressure"]
        if f in df.columns
    ]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=features)

    # --- sample for performance ---
    if len(df) > SAMPLE_SIZE:
        df = df.tail(SAMPLE_SIZE).reset_index(drop=True)

    print(f"✅ Loaded {len(df)} rows | Features: {features} | Time column: {time_col}")
    print(df[["time"] + features].head(3))
    return df, features


# ------------------------------------------------------------
# 2. SYNTHETIC ANOMALY GENERATION
# ------------------------------------------------------------
def inject_anoms(df_in, features, n_points=150, strength=4.0):
    df_out = df_in.copy()
    df_out["gt_anomaly"] = 0
    idxs = np.random.choice(len(df_out), size=min(n_points, len(df_out)), replace=False)
    sigma = df_out[features].std().replace(0, 1e-6)

    for i in idxs:
        cols = np.random.choice(features, size=np.random.randint(1, len(features)), replace=False)
        for c in cols:
            df_out.at[i, c] += np.random.choice([+1, -1]) * strength * sigma[c]
        df_out.at[i, "gt_anomaly"] = 1
    return df_out


# ------------------------------------------------------------
# 3. TRAIN MODELS
# ------------------------------------------------------------
def train_models(train_df, features):
    X_train = train_df[features].values

    if_model = IsolationForest(contamination=CONTAMINATION, random_state=42, n_estimators=200)
    if_model.fit(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    ocsvm = OneClassSVM(kernel="rbf", nu=NU_VAL, gamma="scale")
    ocsvm.fit(X_scaled)

    print("🤖 Models trained successfully.")
    return if_model, ocsvm, scaler


# ------------------------------------------------------------
# 4. REAL-TIME STREAM SIMULATION
# ------------------------------------------------------------
def stream_data(stream_df, features, if_model, ocsvm, scaler):
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print(f"📡 Starting real-time stream simulation ({len(stream_df)} records)...")
    all_data = []

    for i, row in stream_df.iterrows():
        x = row[features].values.reshape(1, -1)

        if_pred = if_model.predict(x)[0]
        if_flag = 1 if if_pred == -1 else 0

        x_scaled = scaler.transform(x)
        oc_pred = ocsvm.predict(x_scaled)[0]
        oc_flag = 1 if oc_pred == -1 else 0

        result = {
            "Index": i,
            "Time": row["time"],
            **{f: row[f] for f in features},
            "IF_Flag": if_flag,
            "OC_Flag": oc_flag,
            "GT_Label": int(row.get("gt_anomaly", 0)),
        }
        all_data.append(result)

        if i % 10 == 0 or i == len(stream_df) - 1:
            pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)
            total_anoms = sum([r["IF_Flag"] or r["OC_Flag"] for r in all_data])
            print(f"Stream {i}/{len(stream_df)} | Detected anomalies: {total_anoms}")

        time.sleep(STREAM_DELAY)

    print("✅ Streaming finished.")
    print(f"💾 Log saved to {OUTPUT_FILE}")


# ------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------
def main():
    print("🚀 IoT Weather Anomaly Detection (Real-time Stream)")
    df, features = load_data()

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    stream_df = df.iloc[split_idx:].reset_index(drop=True)

    if SYNTHETIC:
        stream_df = inject_anoms(stream_df, features, SYNTH_POINTS)
        print(f"🔬 Injected {stream_df['gt_anomaly'].sum()} synthetic anomalies.")

    if_model, ocsvm, scaler = train_models(train_df, features)
    stream_data(stream_df, features, if_model, ocsvm, scaler)


if __name__ == "__main__":
    main()
