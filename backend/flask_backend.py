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

# Load YOLO model (Make sure to replace this with your actual model initialization)
yolo_model = YOLO("backend/vegetable.pt")  # TODO: Load your YOLO model here

# Function to get food alternative based on dataset
def get_alternative(food_name):
    food_name = food_name.strip().lower()
    if food_name not in data['Food Name'].str.lower().values:
        return {'error': f"'{food_name}' is not in the dataset! No substitute found."}
    
    substitute = data.loc[data['Food Name'].str.lower() == food_name, 'Alternative'].values
    if len(substitute) > 0:
        return {'ingredient': food_name, 'substitute':str(substitute)}
    else:
        return {'error': 'No substitute found.'}

@app.route('/substitute', methods=['POST'])
def get_substitutes():
    try:
        data = request.json
        ingredient = data.get('ingredient')
        if not data or 'ingredient' not in data:
            return jsonify({'error': 'Ingredient is required'}), 400

        ingredient = data['ingredient'].strip().lower()
        result = get_alternative(ingredient)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 🔹 Updated /predict endpoint to handle IMAGE uploads
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

        # Get detected vegetable name
        detected_vegetable = results[0].names[int(results[0].boxes.cls[0])]

        # Find a substitute for detected vegetable
        substitute_response = get_alternative(detected_vegetable.lower())
        print({'detected_vegetable': detected_vegetable, **substitute_response})

        return jsonify({
    'detected_vegetable': detected_vegetable,
    'substitute': substitute_response["substitute"]  # Ensure it's a proper list
})


    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)