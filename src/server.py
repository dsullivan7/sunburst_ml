import asyncio
import io
import os
import numpy as np
import tensorflow as tf
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

# model = xgb.Booster()
# model.load_model(os.path.join(dirname, "model/sunburst_ml.json"))

model = tf.keras.models.load_model(os.path.join(dirname, "model/lstm_price_predictor.keras"))

with open(os.path.join(dirname, "model/train_scaler_lstm.pkl"), "rb") as file:
  train_scaler = load(file)
with open(os.path.join(dirname, "model/target_scaler_lstm.pkl"), "rb") as file:
  target_scaler = load(file)

@app.get("/")
def get_health():
   return {"response": "Welcome to the Sunburst Energy API"}

@app.get("/predictions")
async def get_prediction(time_stamp: str):
  date = pd.to_datetime(time_stamp)
  date_minus_1 = date - pd.Timedelta(hours=1)
  date_minus_2 = date - pd.Timedelta(hours=2)
  date_minus_3 = date - pd.Timedelta(hours=3)
  date_minus_4 = date - pd.Timedelta(hours=4)
  date_minus_5 = date - pd.Timedelta(hours=5)
  date_minus_6 = date - pd.Timedelta(hours=6)
  date_minus_7 = date - pd.Timedelta(hours=7)
  date_minus_8 = date - pd.Timedelta(hours=8)
  date_minus_9 = date - pd.Timedelta(hours=9)
  date_minus_10 = date - pd.Timedelta(hours=10)
  date_minus_11 = date - pd.Timedelta(hours=11)
  date_minus_12 = date - pd.Timedelta(hours=12)
  date_minus_13 = date - pd.Timedelta(hours=13)
  date_minus_14 = date - pd.Timedelta(hours=14)
  date_minus_15 = date - pd.Timedelta(hours=15)
  date_minus_16 = date - pd.Timedelta(hours=16)
  date_minus_17 = date - pd.Timedelta(hours=17)
  date_minus_18 = date - pd.Timedelta(hours=18)
  date_minus_19 = date - pd.Timedelta(hours=19)
  date_minus_20 = date - pd.Timedelta(hours=20)
  date_minus_21 = date - pd.Timedelta(hours=21)
  date_minus_22 = date - pd.Timedelta(hours=22)
  date_minus_23 = date - pd.Timedelta(hours=23)

  (
     nyiso_data,
     nyiso_data_minus_1,
     nyiso_data_minus_2,
     nyiso_data_minus_3,
     nyiso_data_minus_4,
     nyiso_data_minus_5,
     nyiso_data_minus_6,
     nyiso_data_minus_7,
     nyiso_data_minus_8,
     nyiso_data_minus_9,
     nyiso_data_minus_10,
     nyiso_data_minus_11,
     nyiso_data_minus_12,
     nyiso_data_minus_13,
     nyiso_data_minus_14,
     nyiso_data_minus_15,
     nyiso_data_minus_16,
     nyiso_data_minus_17,
     nyiso_data_minus_18,
     nyiso_data_minus_19,
     nyiso_data_minus_20,
     nyiso_data_minus_21,
     nyiso_data_minus_22,
     nyiso_data_minus_23,
  ) = await asyncio.gather(
     nyiso.get_data(date),
     nyiso.get_data(date_minus_1),
     nyiso.get_data(date_minus_2),
     nyiso.get_data(date_minus_3),
     nyiso.get_data(date_minus_4),
     nyiso.get_data(date_minus_5),
     nyiso.get_data(date_minus_6),
     nyiso.get_data(date_minus_7),
     nyiso.get_data(date_minus_8),
     nyiso.get_data(date_minus_9),
     nyiso.get_data(date_minus_10),
     nyiso.get_data(date_minus_11),
     nyiso.get_data(date_minus_12),
     nyiso.get_data(date_minus_13),
     nyiso.get_data(date_minus_14),
     nyiso.get_data(date_minus_15),
     nyiso.get_data(date_minus_16),
     nyiso.get_data(date_minus_17),
     nyiso.get_data(date_minus_18),
     nyiso.get_data(date_minus_19),
     nyiso.get_data(date_minus_20),
     nyiso.get_data(date_minus_21),
     nyiso.get_data(date_minus_22),
     nyiso.get_data(date_minus_23),
  )

  data = train_scaler.transform(list(map(lambda x: [
      x.nyc_parameters.year,
      x.nyc_parameters.month,
      x.nyc_parameters.day,
      x.nyc_parameters.minutes,
      x.nyc_parameters.max_temp,
      x.nyc_parameters.min_temp,
      x.nyc_parameters.max_wet_bulb,
      x.nyc_parameters.min_wet_bulb,
      x.nyc_parameters.day_ahead_price,
      x.nyc_parameters.load_forecast,
      x.nyc_parameters.day_of_week,
      x.nyc_parameters.is_holiday,
     ],
     [
        nyiso_data,
        nyiso_data_minus_1,
        nyiso_data_minus_2,
        nyiso_data_minus_3,
        nyiso_data_minus_4,
        nyiso_data_minus_5,
        nyiso_data_minus_6,
        nyiso_data_minus_7,
        nyiso_data_minus_8,
        nyiso_data_minus_9,
        nyiso_data_minus_10,
        nyiso_data_minus_11,
        nyiso_data_minus_12,
        nyiso_data_minus_13,
        nyiso_data_minus_14,
        nyiso_data_minus_15,
        nyiso_data_minus_16,
        nyiso_data_minus_17,
        nyiso_data_minus_18,
        nyiso_data_minus_19,
        nyiso_data_minus_20,
        nyiso_data_minus_21,
        nyiso_data_minus_22,
        nyiso_data_minus_23,
      ]
    )))

  # dmatrix = xgb.DMatrix(data_scaled)
  # results_scaled = model.predict(dmatrix)
  # result = target_scaler.inverse_transform(results_scaled.reshape(-1, 1))[0][0].astype(float)

  result_scaled = model.predict(np.array(data).reshape(1, 24, -1))
  result = target_scaler.inverse_transform(result_scaled)[0][0].astype(float)

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