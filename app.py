# Flask app.py (provided earlier)

from flask import Flask, request, render_template, send_from_directory
import os
from utils import process_video
from ultralytics import YOLO
import torch
from model import TrafficLSTM

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load models once
yolo_model = YOLO("yolov11_weights.pt")
lstm_model = TrafficLSTM()
lstm_model.load_state_dict(torch.load("lstm_traffic.pt", map_location=torch.device("cpu")))
lstm_model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(filepath)

        output_path = process_video(filepath, yolo_model, lstm_model)
        return render_template("index.html", result_video=os.path.basename(output_path))

    return render_template("index.html")

@app.route('/video/<filename>')
def video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)
