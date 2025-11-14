# ============================================================
# COMP7707 A3 - Real-time IoT Analytics System Prototype (Final)
# Member A - System Design & Implementation
# Dataset: SGSC_Weather_Sensor_Data.csv (Time = YYYYMMDDHHMMSS)
# ============================================================

import os
import time
import urllib.request
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 0) CONFIG
# -----------------------------
DATA_URL = (
    "https://data.gov.au/data/dataset/"
    "southern-grampians-weather-sensor-data/resource/"
    "82a5e953-00dc-42d6-9c07-3066bf800be3/"
    "download/SGSC_Weather_Sensor_Data.csv"
)

# Local raw-data cache + streaming output for Streamlit dashboard
LOCAL_PATH = "SGSC_Weather_Sensor_Data.csv"
OUTPUT_FILE = "stream_output.csv"

# Time-ordered split + model hyperparameters
TRAIN_RATIO = 0.7
CONTAMINATION = 0.05  # Isolation Forest
AE_EPOCHS = 10
AE_BATCH_SIZE = 256
AE_LR = 1e-3

# Streaming behaviour (simulation)
STREAM_DELAY = 0.2      # seconds/record
SAMPLE_SIZE = 10000     # limit rows for faster demo

# Synthetic anomaly injection for evaluation
SYNTHETIC = True
SYNTH_POINTS = 150

# Reproducibility
RANDOM_SEED = 42
YEAR_START = 2018
YEAR_END = 2021

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) UTILS
# -----------------------------
def _coerce_time_yyyymmddhhmmss(series: pd.Series) -> pd.Series:
    """
    Parse a time-like column into pandas datetime using YYYYMMDDHHMMSS.

    Handles:
      - plain strings
      - integer timestamps
      - float / exponent notation (e.g. 2.01808E+13)
    """
    s = series.astype(str).str.strip()

    def to_14_digits(x: str) -> str:
        x = x.strip()
        try:
            f = float(x)
            if math.isnan(f):
                return ""
            xi = int(round(f))
            return str(xi)
        except Exception:
            # keep original (may already be clean)
            return x

    # Normalise anything that isn't already exactly 14 digits
    need_fix = ~s.str.fullmatch(r"\d{14}")
    if need_fix.any():
        s.loc[need_fix] = s.loc[need_fix].apply(to_14_digits)

    # Remove non-digits then require length 14
    s = s.str.replace(r"\D", "", regex=True)
    s = s.where(s.str.len() == 14, None)

    dt = pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")
    return dt


def _pick_features(df: pd.DataFrame) -> list:
    """
    Select numeric sensor features for anomaly detection.

    Prefer named weather sensor columns; otherwise fall back to generic
    numeric columns (excluding IDs / geo fields).
    """
    prefer = [
        "airtemp",
        "relativehumidity",
        "windspeed",
        "solar",
        "vapourpressure",
        "atmosphericpressure",
        "gustspeed",
        "winddirection",
    ]
    feats = [c for c in prefer if c in df.columns]

    if len(feats) == 0:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        blacklist = {"deviceid", "eventid", "locationid", "latitude", "longitude"}
        feats = [c for c in num_cols if c not in blacklist and c != "time"]
        feats = feats[:8]  # keep it manageable

    return feats


# -----------------------------
# 2) LOAD DATA
# -----------------------------
def load_data():
    """
    Load dataset (download if missing), parse time, filter by year,
    select features, and clean missing values.
    """
    if not os.path.exists(LOCAL_PATH):
        print("⬇️ Downloading dataset from data.gov.au ...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
        print("✅ Download complete!")

    df = pd.read_csv(LOCAL_PATH, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    print(f"🧾 Columns detected: {list(df.columns)}")

    # Locate time/timestamp column
    time_col = next((c for c in df.columns if "time" in c or "timestamp" in c), None)
    if not time_col:
        raise ValueError("❌ No time/timestamp column found in dataset!")

    if time_col != "time":
        df.rename(columns={time_col: "time"}, inplace=True)

    # Parse time + sort
    df["time"] = _coerce_time_yyyymmddhhmmss(df["time"])
    before_all = len(df)
    df = df.dropna(subset=["time"]).sort_values("time")

    if len(df) == 0:
        raise ValueError("❌ All rows lost while parsing `time` as YYYYMMDDHHMMSS.")

    # Filter by year range (ensures realistic temporal window)
    df = df[(df["time"].dt.year >= YEAR_START) & (df["time"].dt.year <= YEAR_END)]
    if len(df) == 0:
        raise ValueError(f"❌ No rows within year range {YEAR_START}–{YEAR_END}.")

    print(
        f"🕒 Time sample: {df['time'].iloc[0]}  →  "
        f"{df['time'].iloc[min(5, len(df)-1)]}"
    )
    print(
        f"🕒 Time range (filtered): {df['time'].min()} → {df['time'].max()}  "
        f"(kept {len(df)}/{before_all})"
    )

    # Select numeric sensor features
    features = _pick_features(df)
    if len(features) == 0:
        raise ValueError("❌ No numeric features found to model.")

    # Ensure numeric + simple missing-value handling
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()

    # Downsample for demo (keep earliest part so start near 2018)
    if len(df) > SAMPLE_SIZE:
        df = df.head(SAMPLE_SIZE).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if len(df) == 0:
        nz = df[features].notna().sum() if len(df) else "NA"
        raise ValueError(f"❌ Dataset empty after cleaning! features_notna={nz}")

    print(f"✅ Loaded {len(df)} rows | Features: {features}")
    return df, features


# -----------------------------
# 3) SYNTHETIC ANOMALIES
# -----------------------------
def inject_anoms(df_in, features, n_points=150, strength=4.0):
    """
    Inject synthetic anomalies into the streaming partition
    and mark them with gt_anomaly = 1 for evaluation.
    """
    df_out = df_in.copy()
    df_out["gt_anomaly"] = 0

    if len(df_out) == 0:
        return df_out

    n = min(n_points, len(df_out))
    idxs = np.random.choice(len(df_out), size=n, replace=False)
    sigma = df_out[features].std().replace(0, 1e-6)

    for i in idxs:
        # randomly perturb 1..len(features) dimensions
        cols = np.random.choice(
            features, size=np.random.randint(1, max(2, len(features))), replace=False
        )
        for c in cols:
            df_out.at[i, c] += np.random.choice([+1, -1]) * strength * sigma[c]
        df_out.at[i, "gt_anomaly"] = 1

    return df_out


# -----------------------------
# 4) AUTOENCODER MODEL
# -----------------------------
class Autoencoder(nn.Module):
    """
    Simple fully-connected autoencoder for multivariate sensor data.
    """

    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(X_scaled: np.ndarray):
    """
    Train autoencoder on scaled training data and compute
    a reconstruction-error threshold (mean + 3*std).
    """
    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim).to(DEVICE)

    dataset = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(AE_EPOCHS):
        ep_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch.size(0)
        avg_loss = ep_loss / len(dataset)
        print(f"AE epoch {ep+1}/{AE_EPOCHS} | loss={avg_loss:.6f}")

    # Compute reconstruction error on training set to set threshold
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(DEVICE)
        recon = model(X_tensor)
        mse = ((recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()

    threshold = mse.mean() + 3 * mse.std()
    print(f"AE threshold (mean + 3*std): {threshold:.6f}")

    return model, threshold


# -----------------------------
# 5) TRAIN MODELS
# -----------------------------
def train_models(train_df, features):
    """
    Train Isolation Forest + Autoencoder on the historical window.
    These correspond to the classical tree-based and deep-learning
    models discussed in the algorithm evaluation (Member B).
    """
    X_train = train_df[features].values

    # Isolation Forest (tree-based)
    if_model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_estimators=200,
    )
    if_model.fit(X_train)

    # Autoencoder (works on scaled data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    ae_model, ae_threshold = train_autoencoder(X_scaled)

    print("🤖 Models trained successfully (IF + AE).")
    return if_model, ae_model, scaler, ae_threshold


# -----------------------------
# 6) STREAM (write incrementally for Streamlit)
# -----------------------------
def stream_data(stream_df, features, if_model, ae_model, scaler, ae_threshold):
    """
    Simulate real-time streaming:
      - score each record with IF + Autoencoder
      - append flags + GT label into OUTPUT_FILE for the Streamlit dashboard
    """
    cols = ["Index", "Time"] + features + ["IF_Flag", "AE_Flag", "GT_Label"]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

    print(f"📡 Starting real-time stream simulation ({len(stream_df)} records)...")

    buf = []
    total_rows = len(stream_df)

    ae_model.eval()

    for j, (_, row) in enumerate(stream_df.iterrows(), start=1):
        x = row[features].values.reshape(1, -1)

        # Isolation Forest
        if_flag = 1 if if_model.predict(x)[0] == -1 else 0

        # Autoencoder (on scaled data)
        x_scaled = scaler.transform(x).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled).to(DEVICE)

        with torch.no_grad():
            recon = ae_model(x_tensor)
            mse = ((recon - x_tensor) ** 2).mean().item()
        ae_flag = 1 if mse > ae_threshold else 0

        record = {
            "Index": j - 1,
            "Time": row["time"],  # pandas will write ISO string to CSV
            **{f: row.get(f, np.nan) for f in features},
            "IF_Flag": if_flag,
            "AE_Flag": ae_flag,
            "GT_Label": int(row.get("gt_anomaly", 0)),
        }
        buf.append(record)

        # Flush every 10 rows to reduce file contention
        if (j % 10 == 0) or (j == total_rows):
            pd.DataFrame(buf).to_csv(
                OUTPUT_FILE, mode="a", header=False, index=False
            )
            chunk_anoms = sum(
                (r["IF_Flag"] == 1) or (r["AE_Flag"] == 1) for r in buf
            )
            print(
                f"Stream {j}/{total_rows} | "
                f"Detected anomalies in last chunk: {chunk_anoms}"
            )
            buf.clear()

        time.sleep(STREAM_DELAY)

    print("✅ Streaming finished.")
    print(f"💾 Output saved to {OUTPUT_FILE}")


# -----------------------------
# 7) MAIN
# -----------------------------
def main():
    print("🚀 IoT Weather Anomaly Detection (Real-time Stream)")
    df, features = load_data()

    # Time-ordered split to avoid leakage (simulate “past” vs “future”)
    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split]
    stream_df = df.iloc[split:].reset_index(drop=True)

    # Optional synthetic anomalies for evaluation (GT_Label)
    if SYNTHETIC:
        stream_df = inject_anoms(stream_df, features, SYNTH_POINTS)
        print(f"🔬 Injected {int(stream_df['gt_anomaly'].sum())} synthetic anomalies.")

    if_model, ae_model, scaler, ae_threshold = train_models(train_df, features)
    stream_data(stream_df, features, if_model, ae_model, scaler, ae_threshold)


if __name__ == "__main__":
    main()
