# app/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import json, time, numpy as np, torch, joblib, warnings
import paho.mqtt.client as mqtt
from collections import deque
from torch import nn

warnings.filterwarnings("ignore")  # Hide library warnings

# -----------------------------
# Load trained CNN model
# -----------------------------
class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

try:
    label_map = joblib.load("models/label_map.pkl")
    inv_label_map = {v: k for k, v in label_map.items()}
    model = TinyCNN(len(label_map))
    model.load_state_dict(torch.load("models/ecg_cnn.pth", map_location="cpu"))
    model.eval()
    model_ready = True
except Exception as e:
    model_ready = False
    st.warning(f"⚠️ Model not loaded: {e}")

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="IoT Cardiology Dashboard", layout="wide")
st.title("💓 IoT-Based Smart Cardiology Dashboard")
st.markdown("Real-time ECG signal monitoring and AI-driven cardiac insights.")

placeholder_chart = st.empty()
placeholder_metrics = st.empty()
placeholder_pred = st.empty()
placeholder_alert = st.empty()

# Buffers for live data
ecg_buffer = deque([0.0] * 140, maxlen=140)
hr, spo2, temp = 0, 0, 0
prediction = "Waiting for data..."

# -----------------------------
# MQTT Setup
# -----------------------------
BROKER = "broker.hivemq.com"
TOPIC = "iot/cardiology/sim"

def on_connect(client, userdata, flags, rc):
    client.subscribe(TOPIC)
    st.sidebar.success("Connected to MQTT Broker ✅")

def on_message(client, userdata, msg):
    global ecg_buffer, hr, spo2, temp, prediction
    try:
        data = json.loads(msg.payload.decode())
        hr = data.get("heart_rate", 0)
        spo2 = data.get("spo2", 0)
        temp = data.get("temperature", 0)
        ecg = data.get("ecg", [])
        if ecg:
            ecg_buffer.clear()
            ecg_buffer.extend(ecg)

        # AI Prediction (if model loaded)
        if model_ready and ecg:
            x = torch.tensor(np.array(ecg, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                pred = model(x).argmax(1).item()
                prediction = inv_label_map.get(pred, "Unknown")
    except Exception as e:
        st.error(f"Data Error: {e}")

# Initialize MQTT Client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_start()

st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.write("Data Source: MQTT → Topic `iot/cardiology/sim`")
st.sidebar.caption("Run `simulate_iot.py` to stream data")

# -----------------------------
# Real-time Update Loop
# -----------------------------
while True:
    # ECG Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(ecg_buffer), mode="lines", name="ECG"))
    fig.update_layout(title="Real-time ECG Signal", xaxis_title="Time", yaxis_title="Amplitude")
    placeholder_chart.plotly_chart(fig, use_container_width=True, key=time.time())  # Unique key fixes duplicate ID bug

    # Metrics (live vitals)
    col1, col2, col3 = placeholder_metrics.columns(3)
    col1.metric("Heart Rate (BPM)", hr)
    col2.metric("SpO₂ (%)", spo2)
    col3.metric("Temperature (°C)", temp)

    # AI Prediction
    placeholder_pred.markdown(f"### 🩺 Prediction: **{prediction}**")

    # Alert conditions
    if hr > 100 or spo2 < 94 or temp > 37:
        placeholder_alert.error("⚠️ Abnormal vitals detected! Check patient immediately.")
    else:
        placeholder_alert.success("✅ Vitals Normal")

    # Sleep between updates
    time.sleep(2)
