import joblib
import pandas as pd
import numpy as np
import os
import socket
from datetime import datetime
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

# Determine base directory where app.py and index.html live
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app to serve static files from BASE_DIR
app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

# Load model
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df["weekday"] = df["weekday"].astype(int)

latest_preds = []

def compute_predictions():
    now = datetime.now()
    bin_min = (now.minute // 10) * 10
    time_str = f"{now.hour:02d}:{bin_min:02d}"
    weekday = now.weekday()

    subset = df[(df["time_str"] == time_str) & (df["weekday"] == weekday)]
    results = []
    for _, row in subset.iterrows():
        feat = np.array([[
            row["ft_5"], row["ft_4"], row["ft_3"],
            row["ft_2"], row["ft_1"],
            row["lat"], row["lon"],
            row["weekday"], row["exp_avg"]
        ]])
        pred = model.predict(feat)[0]
        results.append({
            "cluster_id": int(row["cluster_id"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "predicted_demand": int(round(pred))
        })
    global latest_preds
    latest_preds = results

# Schedule background updates
scheduler = BackgroundScheduler()
scheduler.add_job(compute_predictions, "interval", minutes=10)
scheduler.start()
compute_predictions()

@app.route('/')
def index():
    # Serve index.html from static folder
    return app.send_static_file('index.html')

@app.route('/api/predictions')
def api_predictions():
    time_str_q = request.args.get('time')
    weekday_q = request.args.get('weekday', type=int)
    if time_str_q and weekday_q is not None:
        subset = df[(df['time_str'] == time_str_q) & (df['weekday'] == weekday_q)]
        results = []
        for _, row in subset.iterrows():
            feat = np.array([[
                row['ft_5'], row['ft_4'], row['ft_3'],
                row['ft_2'], row['ft_1'],
                row['lat'], row['lon'],
                row['weekday'], row['exp_avg']
            ]])
            pred = model.predict(feat)[0]
            results.append({
                'cluster_id': int(row['cluster_id']),
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'predicted_demand': int(round(pred))
            })
        return jsonify(results)
    return jsonify(latest_preds)

@app.route('/favicon.ico')
def favicon():
    return ('', 204)


def find_free_port(start=5000, end=5100):
    for port in range(start, end+1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('0.0.0.0', port)) != 0:
                return port
    raise RuntimeError(f"No free ports available between {start} and {end}")

if __name__ == '__main__':
    port = find_free_port()
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, use_reloader=False)