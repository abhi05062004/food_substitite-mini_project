import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def load_label_encoder(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label encoder file not found: {file_path}")
    with open(file_path, 'rb') as file:
        encoder = pickle.load(file)
    
    # Ensure it's a LabelEncoder object
    if not isinstance(encoder, LabelEncoder):
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(encoder)  # Assuming it's stored as an array
        return label_encoder
    
    return encoder

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)

# Load the model and encoders
model_path = 'D:\\samplefor changes\\backend\\food_substitution_model.h5'
food_encoder_path = 'D:\\samplefor changes\\backend\\label_encoder_food.pkl'
alt_encoder_path = 'D:\\samplefor changes\\backend\\label_encoder_alt.pkl'

model = load_model(model_path)
food_encoder = load_label_encoder(food_encoder_path)
alt_encoder = load_label_encoder(alt_encoder_path)

def get_food_alternative(user_input):
    if user_input not in food_encoder.classes_:
        return "Food item not found in database."
    
    food_index = food_encoder.transform([user_input])
    prediction = model.predict(np.array([food_index]))
    alternative_index = np.argmax(prediction)
    alternative_food = alt_encoder.inverse_transform([alternative_index])[0]
    
    return alternative_food

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a food item: ")
    alternative = get_food_alternative(user_input)
    print(f"Suggested alternative: {alternative}")
