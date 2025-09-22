import joblib

def predict_data(X):
    """
    Predict using Decision Tree model
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    return y_pred


def predict_rf(X):
    """
    Predict using Random Forest model
    """
    model = joblib.load("../model/rf_model.pkl")
    y_pred = model.predict(X)
    return y_pred


def predict_lr(X):
    """
    Predict using Logistic Regression model
    """
    model = joblib.load("../model/lr_model.pkl")
    y_pred = model.predict(X)
    return y_pred
