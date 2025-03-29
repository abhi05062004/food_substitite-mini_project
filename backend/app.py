import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Define correct file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "food_alternatives_dataset.csv")
model_path = os.path.join(BASE_DIR, "food_substitution_model.h5")

# Check if files exist
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file not found: {file_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load dataset
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Load trained food substitution model
model = tf.keras.models.load_model(model_path)

# Set up image upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
yolo_model = YOLO("backend/vegetable.pt")  # Replace with correct path

# Function to get food alternative based on dataset
def get_alternative(food_name):
    food_name = food_name.strip().lower()
    
    # Ensure dataset food names are lowercase
    data["Food Name"] = data["Food Name"].str.lower()
    
    if food_name not in data["Food Name"].values:
        return {'error': f"'{food_name}' is not in the dataset! No substitute found."}

    substitute = data.loc[data["Food Name"] == food_name, "Alternative"].values
    if len(substitute) > 0:
        return {'ingredient': food_name, 'substitute': substitute[0]}
    else:
        return {'error': 'No substitute found.'}

@app.route('/substitute', methods=['POST'])
def get_substitutes():
    try:
        data = request.json
        ingredient = data.get('ingredient', '').strip().lower()
        
        if not ingredient:
            return jsonify({'error': 'Ingredient is required'}), 400

        result = get_alternative(ingredient)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_vegetable():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)

    # Ensure only images are uploaded
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload an image (JPG, PNG, JPEG)'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Read and process image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO

        # Run YOLO model for vegetable detection
        if yolo_model is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500
        
        results = yolo_model(img)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return jsonify({'error': 'No vegetable detected'}), 404

        # Handle multiple detections (return all detected items)
        detected_vegetables = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected_vegetables.append(results[0].names.get(class_id, "Unknown"))

        # Find substitutes for all detected vegetables
        substitutes = [get_alternative(veg.lower()) for veg in detected_vegetables]

        return jsonify({'detected_vegetables': detected_vegetables, 'substitutes': substitutes})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
