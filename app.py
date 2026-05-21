%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from scipy.signal import savgol_filter

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="V2 Jet Engine Intelligence Dashboard", layout="wide")
MAX_LIFE = 125
SEQUENCE_LENGTH = 50

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    reg_model = load_model('rul_regressor_v2.h5', compile=False)
    class_model = load_model('rul_classifier_v2.h5', compile=False)
    scaler = joblib.load('scaler_v2.pkl')
    return reg_model, class_model, scaler

# Load data
columns = ['unit_id', 'cycles', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=columns)
true_rul = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
sensor_cols = test_df.columns.drop(['unit_id', 'cycles'])

reg_model, class_model, scaler = load_assets()

# --- 3. UI LAYOUT ---
st.title("✈️ V2 Jet Engine Intelligence Dashboard")
st.sidebar.header("Engine Control Panel")
selected_engine = st.sidebar.selectbox("Select Target Jet Engine ID:", test_df['unit_id'].unique())

# --- 4. DATA PIPELINE ---
engine_data = test_df[test_df['unit_id'] == selected_engine].copy()
actual_rul = int(true_rul['RUL'].iloc[selected_engine - 1])

# Apply smoothing & scaling
for col in sensor_cols:
    engine_data[col] = savgol_filter(engine_data[col], window_length=11, polyorder=2)
engine_data[sensor_cols] = scaler.transform(engine_data[sensor_cols])

# Sequence Prep
seq = engine_data[sensor_cols].values[-SEQUENCE_LENGTH:]
input_seq = seq.reshape(1, SEQUENCE_LENGTH, len(sensor_cols))

# Inference
pred_rul = float(reg_model.predict(input_seq, verbose=0)[0][0])
fail_prob = float(class_model.predict(input_seq, verbose=0)[0][0])

# --- 5. DASHBOARD METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Predicted RUL", f"{max(0.0, min(125.0, pred_rul)):.1f} Cycles")
col2.metric("Actual RUL", f"{actual_rul} Cycles")
col3.metric("Failure Risk (30d)", f"{fail_prob*100:.1f}%")

if fail_prob >= 0.5:
    st.error("🚨 CRITICAL: Failure imminent within 30 cycles!")
else:
    st.success("🟢 STATUS: Operational")
