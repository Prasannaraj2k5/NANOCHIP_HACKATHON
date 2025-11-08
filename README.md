# NANOCHIP_HACKATHON
This project enables continuous cardiac health monitoring by connecting a FireBoltt smartwatch to an ESP32 microcontroller, displaying live heart-rate data on an OLED screen, and forwarding the data to a Streamlit AI dashboard for advanced analysis.
The dashboard visualizes:
Real-time ECG waveform (simulated or sensor-based)
Heart Rate (HR), SpO₂, and temperature
AI-based prediction (“Normal”, “Stress”, “High Stress”)
Risk level using ML models (CNN / Autoencoder / XGBoost)

+---------------------+          +-------------------------+
| FireBoltt Smartwatch| BLE ---> | ESP32 DevKit V1         |
| Heart & SpO₂ Sensor |          | BLE Client + OLED (SSD1306) |
+---------------------+          | Publishes data to MQTT  |
                                 +-----------+-------------+
                                             |
                                             | MQTT (HiveMQ / Cloud)
                                             v
                            +-------------------------------------+
                            | Streamlit AI Dashboard (Python)     |
                            | - Live ECG Visualization            |
                            | - AI Prediction (CNN)               |
                            | - Risk Stratification (XGBoost)     |
                            | - Autoencoder Anomaly Detection     |
                            +-------------------------------------+
🧠 Features

✅ Smartwatch Integration:
Connects FireBoltt 100 via BLE to ESP32 to fetch real-time HR data.

✅ OLED Display Dashboard:
Displays live HR, constant SpO₂, and AI health state.

✅ AI-Powered Streamlit App:
Analyzes ECG data using CNN & Autoencoder models.

✅ IoT MQTT Connectivity:
ESP32 publishes data to MQTT broker for real-time dashboard updates.

✅ Risk Prediction:
XGBoost model predicts patient risk level based on live vitals.

✅ Expandable:
Can be extended to store data in Google Sheets / Firebase / AWS.

🧩 Tech Stack
Component	Technology Used
Hardware	ESP32 DevKit V1, FireBoltt Smartwatch, OLED SSD1306
Communication	BLE (Bluetooth Low Energy), MQTT (HiveMQ)
AI Models	CNN, Autoencoder, XGBoost (Python, PyTorch)
Dashboard	Streamlit + Plotly
Cloud/Storage	Optional: Firebase, Google Sheets
IDE/Tools	Arduino IDE + VS Code + Python (venv)
