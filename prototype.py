# ============================================================
# COMP7707 A3 - Real-time IoT Analytics System Prototype (Final v6)
# Member A - System Design & Implementation
# Dataset: SGSC_Weather_Sensor_Data.csv (Time = YYYYMMDDHHMMSS)
# ============================================================

import os, time, urllib.request, math
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 0) CONFIG
# -----------------------------
DATA_URL = (
    "https://data.gov.au/data/dataset/"
    "southern-grampians-weather-sensor-data/resource/"
    "82a5e953-00dc-42d6-9c07-3066bf800be3/"
    "download/SGSC_Weather_Sensor_Data.csv"
)
LOCAL_PATH   = "SGSC_Weather_Sensor_Data.csv"   # ใช้ไฟล์จริงที่คุณมี
OUTPUT_FILE  = "stream_output.csv"

TRAIN_RATIO   = 0.7
CONTAMINATION = 0.05
NU_VAL        = 0.05
STREAM_DELAY  = 0.2      # วินาที/เรคอร์ด (จำลองการสตรีม)
SAMPLE_SIZE   = 10000    # จำกัดจำนวนแถวเพื่อเดโมเร็ว
SYNTHETIC     = True
SYNTH_POINTS  = 150
RANDOM_SEED   = 42

YEAR_START = 2018
YEAR_END   = 2021

np.random.seed(RANDOM_SEED)

# -----------------------------
# 1) UTILS
# -----------------------------
def _coerce_time_yyyymmddhhmmss(series: pd.Series) -> pd.Series:
    """
    แปลงคอลัมน์เวลาเป็น datetime จากรูปแบบ YYYYMMDDHHMMSS
    รองรับทั้ง string/int/float/exponent (เช่น 2.01808E+13)
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
        except:
            return x

    need_fix = ~s.str.fullmatch(r"\d{14}")
    if need_fix.any():
        s.loc[need_fix] = s.loc[need_fix].apply(to_14_digits)

    s = s.str.replace(r"\D", "", regex=True)
    s = s.where(s.str.len() == 14, None)
    dt = pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")
    return dt


def _pick_features(df: pd.DataFrame) -> list:
    prefer = [
        "airtemp", "relativehumidity", "windspeed", "solar", "vapourpressure",
        "atmosphericpressure", "gustspeed", "winddirection",
    ]
    feats = [c for c in prefer if c in df.columns]
    if len(feats) == 0:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        blacklist = {"deviceid","eventid","locationid","latitude","longitude"}
        feats = [c for c in num_cols if c not in blacklist and c != "time"]
        feats = feats[:8]
    return feats

# -----------------------------
# 2) LOAD DATA
# -----------------------------
def load_data():
    """โหลดไฟล์จริง (ถ้าไม่มีค่อยดาวน์โหลด) และแปลงเวลาถูกต้อง"""
    if not os.path.exists(LOCAL_PATH):
        print("⬇️ Downloading dataset from data.gov.au ...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
        print("✅ Download complete!")

    df = pd.read_csv(LOCAL_PATH, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    print(f"🧾 Columns detected: {list(df.columns)}")

    # หา time column
    time_col = next((c for c in df.columns if "time" in c or "timestamp" in c), None)
    if not time_col:
        raise ValueError("❌ No time/timestamp column found in dataset!")
    if time_col != "time":
        df.rename(columns={time_col: "time"}, inplace=True)

    # แปลงเป็น datetime และเรียงเวลา
    df["time"] = _coerce_time_yyyymmddhhmmss(df["time"])
    before_all = len(df)
    df = df.dropna(subset=["time"]).sort_values("time")
    if len(df) == 0:
        raise ValueError("❌ All rows lost while parsing `time` as YYYYMMDDHHMMSS.")

    # ✅ กรองช่วงปี 2018–2021 แน่นอน
    df = df[(df["time"].dt.year >= YEAR_START) & (df["time"].dt.year <= YEAR_END)]
    if len(df) == 0:
        raise ValueError(f"❌ No rows within year range {YEAR_START}–{YEAR_END}.")
    print(f"🕒 Time sample: {df['time'].iloc[0]}  →  {df['time'].iloc[min(5, len(df)-1)]}")
    print(f"🕒 Time range (filtered): {df['time'].min()} → {df['time'].max()}  (kept {len(df)}/{before_all})")

    # เลือก feature
    features = _pick_features(df)
    if len(features) == 0:
        raise ValueError("❌ No numeric features found to model.")

    # แปลงเป็นตัวเลข + เติมช่องว่าง
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()

    # ✅ ลดขนาดเพื่อเดโมเร็ว: ใช้ข้อมูล "ช่วงต้นสุด" เพื่อให้เริ่มที่ปี 2018 จริง
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
    """สุ่ม inject ค่าผิดปกติลงในสตรีมเพื่อทดสอบ"""
    df_out = df_in.copy()
    df_out["gt_anomaly"] = 0
    if len(df_out) == 0:
        return df_out
    n = min(n_points, len(df_out))
    idxs = np.random.choice(len(df_out), size=n, replace=False)
    sigma = df_out[features].std().replace(0, 1e-6)
    for i in idxs:
        cols = np.random.choice(features, size=np.random.randint(1, max(2, len(features))), replace=False)
        for c in cols:
            df_out.at[i, c] += np.random.choice([+1, -1]) * strength * sigma[c]
        df_out.at[i, "gt_anomaly"] = 1
    return df_out

# -----------------------------
# 4) TRAIN MODELS
# -----------------------------
def train_models(train_df, features):
    X_train = train_df[features].values

    if_model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_estimators=200
    )
    if_model.fit(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    ocsvm = OneClassSVM(kernel="rbf", nu=NU_VAL, gamma="scale")
    ocsvm.fit(X_scaled)

    print("🤖 Models trained successfully.")
    return if_model, ocsvm, scaler

# -----------------------------
# 5) STREAM (write incrementally for Streamlit)
# -----------------------------
def stream_data(stream_df, features, if_model, ocsvm, scaler):
    # เขียน header ทีเดียว กัน Streamlit อ่านขณะเขียน
    cols = ["Index", "Time"] + features + ["IF_Flag", "OC_Flag", "GT_Label"]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

    print(f"📡 Starting real-time stream simulation ({len(stream_df)} records)...")

    buf = []  # buffer เขียนเป็นก้อน ๆ ลดโอกาสไฟล์เสียหาย
    total_rows = len(stream_df)
    for j, (_, row) in enumerate(stream_df.iterrows(), start=1):
        x = row[features].values.reshape(1, -1)

        # Isolation Forest
        if_flag = 1 if if_model.predict(x)[0] == -1 else 0
        # One-Class SVM
        x_scaled = scaler.transform(x)
        oc_flag = 1 if ocsvm.predict(x_scaled)[0] == -1 else 0

        record = {
            "Index": j-1,
            "Time": row["time"],  # datetime จริง
            **{f: row.get(f, np.nan) for f in features},
            "IF_Flag": if_flag,
            "OC_Flag": oc_flag,
            "GT_Label": int(row.get("gt_anomaly", 0)),
        }
        buf.append(record)

        # เขียนลงไฟล์ทุก ๆ 10 แถว เพื่อลดกระพริบใน Streamlit
        if (j % 10 == 0) or (j == total_rows):
            pd.DataFrame(buf).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
            chunk_anoms = sum((r["IF_Flag"] == 1) or (r["OC_Flag"] == 1) for r in buf)
            print(f"Stream {j}/{total_rows} | Detected anomalies (chunk): {chunk_anoms}")
            buf.clear()

        time.sleep(STREAM_DELAY)

    print("✅ Streaming finished.")
    print(f"💾 Output saved to {OUTPUT_FILE}")

# -----------------------------
# 6) MAIN
# -----------------------------
def main():
    print("🚀 IoT Weather Anomaly Detection (Real-time Stream)")
    df, features = load_data()

    # แยก train/stream ตามลำดับเวลา (สำคัญมากสำหรับสตรีม)
    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split]
    stream_df = df.iloc[split:].reset_index(drop=True)

    if SYNTHETIC:
        stream_df = inject_anoms(stream_df, features, SYNTH_POINTS)
        print(f"🔬 Injected {stream_df['gt_anomaly'].sum()} synthetic anomalies.")

    if_model, ocsvm, scaler = train_models(train_df, features)
    stream_data(stream_df, features, if_model, ocsvm, scaler)

if __name__ == "__main__":
    main()
