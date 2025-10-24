"""
model_training.py
-----------------
Trains a simple TensorFlow neural network on the Iris dataset.
- Standardizes features using StandardScaler
- Saves trained model and scaler inside /models/
- Skips retraining if model already exists
Designed to run cleanly inside Docker or locally.
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def main(epochs: int = 50):
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Prepare directories
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "my_model.keras")
    scaler_path = os.path.join("models", "scaler.pkl")

    # Skip training if model already exists
    if os.path.exists(model_path):
        print(" Model already exists. Skipping training.")
        return

    print(" Starting model training...")

    # Load Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

    # Evaluate performance
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f" Model trained successfully — Accuracy: {acc:.3f}")

    # Save model and scaler
    model.save(model_path)
    joblib.dump(sc, scaler_path)
    print(f" Model saved to: {model_path}")
    print(f" Scaler saved to: {scaler_path}")

    # Save model summary for reference
    with open(os.path.join("models", "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print(" Model summary written to models/model_summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Iris classifier model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()
    main(epochs=args.epochs)
