- Updated model_training.py with full TensorFlow training pipeline
  • Added data loading, preprocessing using StandardScaler, and model training
  • Saves both trained model (my_model.keras) and scaler (scaler.pkl) to /models/
  • Added reproducibility seeds, skip-retraining logic, and model summary export

- Updated main.py Flask app for serving predictions
  • Loads trained model and scaler
  • Handles /predict POST requests and renders predict.html form
  • Configured to run on host 0.0.0.0, port 4000 for Docker

- Created and configured multi-stage Dockerfile
  • Stage 1: trains the model and saves artifacts
  • Stage 2: serves the Flask API using trained model/scaler
  • Added COPY commands for model and scaler, and mkdir for /models
  • Exposed port 4000, fixed ENV syntax, and used --no-cache-dir installs

- Cleaned requirements.txt to include Flask, TensorFlow, scikit-learn, joblib, and numpy
- Verified the complete workflow inside Docker:
  • Image builds successfully
  • Flask app runs at http://localhost:4000/predict
  • Predictions verified for Setosa, Versicolor, and Virginica classes
