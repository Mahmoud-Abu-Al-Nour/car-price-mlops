#Import the libraries
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
with open('models/car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API")

# Input schema
class CarFeatures(BaseModel):
    Year: int
    Present_Price: float
    Driven_kms: int
    Fuel_Type: int      # 0=CNG, 1=Diesel, 2=Petrol
    Seller_Type: int    # 0=Dealer, 1=Individual
    Transmission: int   # 0=Automatic, 1=Manual
    Owner: int

# Root endpoint
@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(car: CarFeatures):
    features = np.array([[
        car.Year,
        car.Present_Price,
        car.Driven_kms,
        car.Fuel_Type,
        car.Seller_Type,
        car.Transmission,
        car.Owner
    ]])
    
    prediction = model.predict(features)
    
    return {
        "predicted_price": round(float(prediction[0]), 2),
        "currency": "Lakh INR"
    }