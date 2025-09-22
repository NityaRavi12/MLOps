# FastAPI

## 
I have completed the required lab by training and serving a **Decision Tree Classifier** using FastAPI. In addition to this, I extended the work by implementing two more models — **Random Forest** and **Logistic Regression** — to compare their performance.  
For this, I made changes to:
- **`train.py`** → to train and save all three models (`iris_model.pkl`, `rf_model.pkl`, `lr_model.pkl`)  
- **`predict.py`** → to load each model and return predictions  
- **`main.py`** → to add separate API endpoints and map predictions to Iris flower class names  

This setup allows testing and comparing multiple models directly through the FastAPI Swagger UI.

---

##  Changes Made
### 1. `train.py`
- Trains **Decision Tree**, **Random Forest**, and **Logistic Regression** models  
- Saves each model into the `../model/` directory as `.pkl` files  
- Prints training accuracy for each model  

### 2. `predict.py`
- Functions to load and predict using the trained models:  
  - `predict_data()` → Decision Tree  
  - `predict_rf()` → Random Forest  
  - `predict_lr()` → Logistic Regression  

### 3. `main.py`
- FastAPI app with endpoints:  
  - `GET /` → Health check  
  - `POST /predict` → Decision Tree  
  - `POST /predict_rf` → Random Forest  
  - `POST /predict_lr` → Logistic Regression  
- Uses **Pydantic models** for request validation  
- Maps class IDs (`0, 1, 2`) to class names (`Setosa, Versicolor, Virginica`)  

---

##  Running the Project
### 1. Train the Models
```bash
python train.py
