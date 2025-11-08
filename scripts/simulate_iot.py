# scripts/simulate_iot.py
import time, json, random
import paho.mqtt.client as mqtt
import math

BROKER = "broker.hivemq.com"  # public MQTT broker
TOPIC = "iot/cardiology/sim"  # topic to publish simulated vitals

client = mqtt.Client()
client.connect(BROKER, 1883, 60)

def generate_ecg(length=140):
    """Generate a synthetic ECG-like waveform"""
    t = [i / 200 for i in range(length)]
    ecg = [0.6 * math.sin(2 * math.pi * 1.7 * x) + 0.15 * random.gauss(0, 1) for x in t]
    if random.random() < 0.05:  # occasional QRS spike
        i = random.randint(10, length - 10)
        ecg[i:i + 3] = [2.0, -0.5, 0.1]
    return ecg

print("🩺 Starting IoT Cardiology Simulator...")
print("Publishing data to MQTT topic:", TOPIC)
print("Press Ctrl + C to stop.\n")

while True:
    hr = random.randint(60, 100)         # heart rate
    spo2 = random.randint(95, 100)       # oxygen level
    temp = round(random.uniform(36.3, 37.2), 2)  # temperature
    ecg = generate_ecg()

    payload = {
        "heart_rate": hr,
        "spo2": spo2,
        "temperature": temp,
        "ecg": ecg,
        "timestamp": int(time.time() * 1000)
    }

    client.publish(TOPIC, json.dumps(payload))
    print(f"✅ Published: HR={hr} | SpO2={spo2}% | Temp={temp}°C")
    time.sleep(2)
