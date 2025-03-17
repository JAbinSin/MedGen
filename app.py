import os
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from werkzeug.utils import secure_filename
import imghdr
from ultralytics import YOLO
import json
import cv2

# Initialize Flask app
app = Flask(__name__)

# Global variables and configurations
app.config['UPLOAD_FOLDER'] = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
model_path = 'model/yolov8_model.pt'
progress = 0

# Add configuration constants
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Load model and medicine info at startup
model = None
medicine_info = None
team_info = None

def init_app():
    global model, medicine_info, team_info
    if model is None:
        print("Loading YOLO model...")
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    
    if medicine_info is None:
        print("Loading medicine information...")
        with open('medicine_info.json') as f:
            medicine_info = json.load(f)
        print("Medicine information loaded successfully")
    
    if team_info is None:
        print("Loading team information...")
        with open('team_info.json') as f:
            team_info = json.load(f)
        print("Team information loaded successfully")

# Initialize when the first request is received
@app.before_request
def before_first_request():
    init_app()

def generate_filename():
    """Generate a unique filename using the current timestamp."""
    return time.strftime("%Y%m%d_%H%M%S") + ".jpg"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(stream):
    """Validate that uploaded file is a supported image type"""
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    progress = 0

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file type
    file_ext = validate_image(file.stream)
    if not file_ext or file_ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif']:
        return jsonify({'error': 'Invalid image format'}), 400

    try:
        # Create secure filename and save
        filename = secure_filename(f"{int(time.time())}{file_ext}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        progress = 10

        # Validate saved file is readable image
        if cv2.imread(file_path) is None:
            os.remove(file_path)  # Clean up invalid file
            progress = 100
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        # Process image with model
        try:
            results = model.predict(file_path, save=True)
            progress = 70

            # Get prediction image path
            save_dir = Path(results[0].save_dir)
            saved_files = list(save_dir.glob("*.jpg"))

            if not saved_files:
                raise Exception("Model failed to generate prediction image")

            # Move and rename prediction image
            predicted_img_path = os.path.join(PREDICTION_FOLDER, filename)
            os.rename(saved_files[0], predicted_img_path)

            # Extract labels
            labels = list(set([model.names[int(detection.cls)] for detection in results[0].boxes]))
            progress = 100

            time.sleep(2)
            
            if not labels:
                labels = ['No Pill Detected']

            return jsonify({'redirect_url': url_for('result', filename=filename, pred_filename=filename, labels=','.join(labels))})

        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Error processing image'}), 500

    except Exception as e:
        return jsonify({'error': 'Error saving file'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

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

    # Sort medicine_info alphabetically by name
    sorted_medicine_info = dict(sorted(medicine_info.items(), key=lambda x: x[1]['name']))

    return render_template('result.html', 
                         file_url=file_url, 
                         labels=labels, 
                         info_list=info_list, 
                         pred_filename=pred_filename,
                         medicine_info=sorted_medicine_info)

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/menu')
def menu():
    return render_template('menu.html', medicine_info=medicine_info)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html', team_info=team_info)

if __name__ == '__main__':
    init_app()  # Initialize when running directly
    app.run(debug=True)
