import os
import uvicorn
from fastapi import FastAPI
import pandas as pd
from pickle import load
from pydantic import BaseModel

class PredictionParameters(BaseModel):
    year: int
    month: int
    day: int
    minutes: int
    max_temp: float
    min_temp: float
    max_wet_bulb: float
    min_wet_bulb: float
    day_ahead_price: float
    load_forecast: float

app = FastAPI()
dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "rb") as f:
  model = load(f)

@app.get("/")
def get_health():
   return {"response": "Welcome to the Sunburst Energy API"}

@app.post("/predictions")
def create_prediction(prediction_parameters: PredictionParameters):
  results = model.predict(pd.DataFrame([[
    prediction_parameters.year,
    prediction_parameters.month,
    prediction_parameters.day,
    prediction_parameters.minutes,
    prediction_parameters.max_temp,
    prediction_parameters.min_temp,
    prediction_parameters.max_wet_bulb,
    prediction_parameters.min_wet_bulb,
    prediction_parameters.day_ahead_price,
    prediction_parameters.load_forecast,
  ]], columns=["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast"]))
  return {"response": results.tolist()}

def start():
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)