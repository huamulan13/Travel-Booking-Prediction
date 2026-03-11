from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "Travel prediction API"}

@app.post("/predict")
def predict(search_count:int, booking_history:int, price_sensitivity:float):

    data = np.array([[1, search_count, booking_history, price_sensitivity]])

    pred = model.predict(data)

    return {"prediction": int(pred[0])}