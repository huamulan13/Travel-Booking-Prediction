from fastapi import FastAPI
import joblib
import numpy as np
from src.monitor_model import log_prediction

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "Travel prediction API"}

@app.get("/predict")
def predict(search_count: int, booking_history: int, price_sensitivity: float):

    input_data = [[search_count, booking_history, price_sensitivity]]
    prediction = model.predict(input_data)[0]

    log_prediction(input_data, prediction)

    return {"prediction": int(prediction)}

# @app.post("/predict")
# def predict(search_count:int, booking_history:int, price_sensitivity:float):

#     data = np.array([[1, search_count, booking_history, price_sensitivity]])

#     pred = model.predict(data)

#     return {"prediction": int(pred[0])}