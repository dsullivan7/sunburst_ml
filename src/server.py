import asyncio
import io
import os
import uvicorn
from fastapi import FastAPI
import pandas as pd
from pickle import load
from pydantic import BaseModel
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import xgboost as xgb

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

# with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "rb") as f:
#   model = load(f)

model = xgb.Booster()
model.load_model(os.path.join(dirname, "model/sunburst_ml.json"))

with open(os.path.join(dirname, "model/train_scaler.pkl"), "rb") as file:
  train_scaler = load(file)
with open(os.path.join(dirname, "model/target_scaler.pkl"), "rb") as file:
  target_scaler = load(file)

@app.get("/")
def get_health():
   return {"response": "Welcome to the Sunburst Energy API"}

@app.get("/predictions")
async def get_prediction(time_stamp: str):
  date = pd.to_datetime(time_stamp)
  date_prev_day = date - pd.Timedelta(days=1)
  date_prev_week = date - pd.Timedelta(days=7)

  nyiso_data, nyiso_data_prev_day, nyiso_data_prev_week = await asyncio.gather(
     nyiso.get_data(date),
     nyiso.get_data(date_prev_day),
     nyiso.get_data(date_prev_week),
  )

  data = [[
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
    nyiso_data_prev_day.nyc_actual_price,
    nyiso_data_prev_week.nyc_actual_price,
  ]]

  data_scaled = pd.DataFrame(train_scaler.transform(data), columns=[
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
    "Is Holiday",
    "Prev Day Price",
    "Prev Week Price",
  ])

  dmatrix = xgb.DMatrix(data_scaled)
  results_scaled = model.predict(dmatrix)
  result = target_scaler.inverse_transform(results_scaled.reshape(-1, 1))[0][0].astype(float)

  # results = model.predict(pd.DataFrame([[
  #   nyiso_data.nyc_parameters.year,
  #   nyiso_data.nyc_parameters.month,
  #   nyiso_data.nyc_parameters.day,
  #   nyiso_data.nyc_parameters.minutes,
  #   nyiso_data.nyc_parameters.max_temp,
  #   nyiso_data.nyc_parameters.min_temp,
  #   nyiso_data.nyc_parameters.max_wet_bulb,
  #   nyiso_data.nyc_parameters.min_wet_bulb,
  #   nyiso_data.nyc_parameters.day_ahead_price,
  #   nyiso_data.nyc_parameters.load_forecast,
  #   nyiso_data.nyc_parameters.day_of_week,
  #   nyiso_data.nyc_parameters.is_holiday,
  # ]], columns=[
  #   "Year",
  #   "Month",
  #   "Day",
  #   "Minutes",
  #   "Max Temp",
  #   "Min Temp",
  #   "Max Wet Bulb",
  #   "Min Wet Bulb",
  #   "Day Ahead Price",
  #   "Load Forecast",
  #   "Day Of Week",
  #   "Is Holiday"
  # ]))

  return {"response": {
     "nyc_actual_price": nyiso_data.nyc_actual_price,
     "nyc_predicted_price": result,
     "nyc_day_ahead_price": nyiso_data.nyc_parameters.day_ahead_price,
  }}

def start():
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)