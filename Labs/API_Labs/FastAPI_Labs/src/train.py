from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from data import load_data, split_data

def fit_models(X_train, y_train):
    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, "../model/iris_model.pkl")
    print("Decision Tree trained and saved as iris_model.pkl")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=12)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "../model/rf_model.pkl")
    print("Random Forest trained and saved as rf_model.pkl")

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "../model/lr_model.pkl")
    print("Logistic Regression trained and saved as lr_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_models(X_train, y_train)
