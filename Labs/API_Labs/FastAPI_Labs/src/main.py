from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, predict_rf, predict_lr  

app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response: int

iris_classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                     iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)   # Decision Tree
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_rf")
async def predict_with_rf(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                     iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_rf(features)
        class_id = int(prediction[0])
        return {
            "model": "Random Forest",
            "class_id": class_id,
            "class_name": iris_classes[class_id]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_lr")
async def predict_with_lr(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                     iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_lr(features)
        class_id = int(prediction[0])
        return {
            "model": "Logistic Regression",
            "class_id": class_id,
            "class_name": iris_classes[class_id]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
