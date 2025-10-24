from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib
import os

# Correct folders for Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model and scaler from /models/
MODEL_PATH = os.path.join("models", "my_model.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Class labels for the Iris dataset
class_labels = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form

            # Extract and convert inputs
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            # Apply the same standardization as during training
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)
            predicted_class = class_labels[np.argmax(prediction)]

            return jsonify({"predicted_class": predicted_class})

        except Exception as e:
            return jsonify({"error": str(e)})

    # For GET requests, render the input form
    return render_template('predict.html')


if __name__ == "__main__":
    # host=0.0.0.0 allows access inside Docker
    app.run(debug=True, host='0.0.0.0', port=4000)
