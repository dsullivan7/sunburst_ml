import io
import os
import uvicorn
from fastapi import FastAPI
import pandas as pd
from pickle import load
from pydantic import BaseModel
import urllib.request
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

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

@app.get("/")
def get_health():
   return {"response": "Welcome to the Sunburst Energy API"}

@app.get("/parameters")
def get_parameters():
  price_data_response = urllib.request.urlopen("https://mis.nyiso.com/public/realtime/realtime_zone_lbmp.csv").read()
  price_data = pd.read_csv(io.BytesIO(price_data_response))

  nyc_price_data = price_data[price_data["Name"] == "N.Y.C."]
  actual_price = nyc_price_data['LBMP ($/MWHr)'].values[0]
  date = pd.to_datetime(nyc_price_data["Time Stamp"].values[0])
  yesterday = date - pd.Timedelta(days=1)

  day_ahead_price_data_response = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/damlbmp/{date.year}{date.month:02d}{date.day:02d}damlbmp_zone.csv").read()
  weather_data_tesponse = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/lfweather/{yesterday.year}{yesterday.month:02d}{(yesterday.day):02d}lfweather.csv").read()
  load_forecast_data_response = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/isolf/{date.year}{date.month:02d}{date.day:02d}isolf.csv").read()

  day_ahead_price_data = pd.read_csv(io.BytesIO(day_ahead_price_data_response))
  day_ahead_price_data["Time Stamp"] = pd.to_datetime(day_ahead_price_data["Time Stamp"])
  nyc_day_ahead_price_data = day_ahead_price_data[day_ahead_price_data["Name"] == "N.Y.C."]
  nyc_day_ahead_price = nyc_day_ahead_price_data[pd.to_datetime(nyc_day_ahead_price_data["Time Stamp"]).dt.hour == date.hour]['LBMP ($/MWHr)'].values[0]

  weather_data = pd.read_csv(io.BytesIO(weather_data_tesponse))
  nyc_weather_data = weather_data[:][(weather_data["Station ID"] == "NYC") & (pd.to_datetime(weather_data["Vintage Date"]) == (pd.to_datetime(weather_data["Forecast Date"]) + pd.Timedelta(days=1)))]

  load_forecast_data = pd.read_csv(io.BytesIO(load_forecast_data_response))
  load_forecast_data["Time Stamp"] = pd.to_datetime(load_forecast_data["Time Stamp"])
  nyc_load_forecast = load_forecast_data[
     (load_forecast_data["Time Stamp"].dt.year == date.year) &
     (load_forecast_data["Time Stamp"].dt.month == date.month) &
     (load_forecast_data["Time Stamp"].dt.day == date.day) &
     (load_forecast_data["Time Stamp"].dt.hour == date.hour)
  ]["N.Y.C."].values[0]

  cal = calendar()
  holidays = cal.holidays(start=date, end=date)

  return {"response": {
    "actual_price": actual_price,
    "time_stamp": date,
    "parameters": PredictionParameters(
      year=date.year,
      month=date.month,
      day=date.day,
      minutes=date.hour * 60 + date.minute,
      max_temp=nyc_weather_data["Max Temp"].values[0],
      min_temp=nyc_weather_data["Min Temp"].values[0],
      max_wet_bulb=nyc_weather_data["Max Wet Bulb"].values[0],
      min_wet_bulb=nyc_weather_data["Min Wet Bulb"].values[0],
      day_ahead_price=nyc_day_ahead_price,
      load_forecast=nyc_load_forecast,
      day_of_week=date.day_of_week,
      is_holiday=date in holidays
    )}
  }

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
    prediction_parameters.day_of_week,
    prediction_parameters.is_holiday,
  ]], columns=["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast", "Day Of Week", "Is Holiday"]))
  return {"response": results.tolist()}

def start():
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)