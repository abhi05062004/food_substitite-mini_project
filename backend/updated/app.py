from flask import Flask, request, jsonify, render_template
import random

app = Flask(__name__)

# Simulated food detection function
def detect_food():
    foods = {"apple": "banana", "milk": "almond milk", "chicken": "tofu"}
    food = random.choice(list(foods.keys()))
    return food, foods[food]

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    food, substitute = detect_food()
    return jsonify({"food": food, "substitute": substitute})

if __name__ == '__main__':
    app.run(debug=True)