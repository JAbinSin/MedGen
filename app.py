import os
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model_path = 'model/yolov8_model.pt'

model = YOLO(model_path)

with open('medicine_info.json') as f:
    medicine_info = json.load(f)

def generate_filename():
    """Generate a unique filename using the current timestamp."""
    return time.strftime("%Y%m%d_%H%M%S") + ".jpg"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

progress = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    progress = 0
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # Generate a unique filename using timestamp
    new_filename = generate_filename()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    
    file.save(file_path)
    progress = 10

    # Perform object detection and save results
    results = model.predict(file_path, save=True)
    progress = 70

    # Get the saved prediction image
    save_dir = Path(results[0].save_dir)
    saved_files = list(save_dir.glob("*.jpg"))

    # Rename and move prediction image
    predicted_img_path = os.path.join(PREDICTION_FOLDER, new_filename)
    if saved_files:
        os.rename(saved_files[0], predicted_img_path)
    else:
        predicted_img_path = file_path  # Fallback to original if no prediction image saved

    # Extract detected class labels
    labels = list(set([model.names[int(detection.cls)] for detection in results[0].boxes]))
    progress = 100

    time.sleep(2)

    # If no objects are detected, set labels to 'No Pill Detected'
    if not labels:
        labels = ['No Pill Detected']

    return jsonify({'redirect_url': url_for('result', filename=new_filename, pred_filename=new_filename, labels=','.join(labels))})

@app.route('/progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify({'progress': progress})

@app.route('/result')
def result():
    filename = request.args.get('filename')
    pred_filename = request.args.get('pred_filename')
    labels = request.args.get('labels', '').split(',')

    file_url = url_for('static', filename='predictions/' + pred_filename) if pred_filename else ''
    info_list = [medicine_info.get(label, {'name': label}) for label in labels]

    return render_template('result.html', file_url=file_url, labels=labels, info_list=info_list, pred_filename=pred_filename)

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/menu')
def menu():
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return redirect(url_for('index'))

@app.route('/contact')
def contact():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
