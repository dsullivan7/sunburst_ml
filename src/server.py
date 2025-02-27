import io
import os
import uvicorn
from fastapi import FastAPI
import pandas as pd
from pickle import load
from pydantic import BaseModel
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from . import nyiso

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
    day_of_week: int
    is_holiday: bool

app = FastAPI()
dirname = os.path.dirname(__file__)

with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "rb") as f:
  model = load(f)

# model = xgb.Booster()
# model.load_model(os.path.join(dirname, "model/sunburst_ml.json"))

# with open(os.path.join(dirname, "model/train_scaler.pkl"), "rb") as file:
#   train_scaler = load(file)
# with open(os.path.join(dirname, "model/target_scaler.pkl"), "rb") as file:
#   target_scaler = load(file)

@app.get("/")
def get_health():
   return {"response": "Welcome to the Sunburst Energy API"}

@app.get("/predictions")
def get_prediction(time_stamp: str):
  nyiso_data = nyiso.get_data(time_stamp)
  results = model.predict(pd.DataFrame([[
    nyiso_data.nyc_parameters.year,
    nyiso_data.nyc_parameters.month,
    nyiso_data.nyc_parameters.day,
    nyiso_data.nyc_parameters.minutes,
    nyiso_data.nyc_parameters.max_temp,
    nyiso_data.nyc_parameters.min_temp,
    nyiso_data.nyc_parameters.max_wet_bulb,
    nyiso_data.nyc_parameters.min_wet_bulb,
    nyiso_data.nyc_parameters.day_ahead_price,
    nyiso_data.nyc_parameters.load_forecast,
    nyiso_data.nyc_parameters.day_of_week,
    nyiso_data.nyc_parameters.is_holiday,
  ]], columns=[
    "Year",
    "Month",
    "Day",
    "Minutes",
    "Max Temp",
    "Min Temp",
    "Max Wet Bulb",
    "Min Wet Bulb",
    "Day Ahead Price",
    "Load Forecast",
    "Day Of Week",
    "Is Holiday"
  ]))

  return {"response": {
     "nyc_actual_price": nyiso_data.nyc_actual_price,
     "nyc_predicted_price": results[0],
     "nyc_day_ahead_price": nyiso_data.nyc_parameters.day_ahead_price,
  }}

def start():
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)