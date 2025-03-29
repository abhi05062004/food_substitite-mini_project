import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load LabelEncoder classes
encoder_classes = np.load("backend/label_encoder_classes.npy", allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = encoder_classes

# Load trained model
substitution_model = tf.keras.models.load_model("backend/food_substitution_model.h5")

# Function to get food alternative
def get_alternative(food_name):
    if food_name not in encoder.classes_:
        return f"'{food_name}' is not in the dataset! No substitute found."

    # Convert food name to numerical ID
    food_id = encoder.transform([food_name])[0]

    # Predict alternative food
    predictions = substitution_model.predict(np.array([food_id]).reshape(1, 1))

    # Get the most probable alternative
    alternative_id = np.argmax(predictions)

    # Get alternative food name
    alternative_food = encoder.inverse_transform([alternative_id])[0]

    return f"Substitute for {food_name}: {alternative_food}"

# Example usage
while True:
    food_input = input("Enter a food name: ").strip().lower()
    result = get_alternative(food_input)
    print(result)
