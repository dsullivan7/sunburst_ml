import calendar
import io
import urllib.request

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from dataclasses import dataclass

@dataclass
class NYISOParameters:
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

@dataclass
class NYISOData:
  nyc_actual_price: float
  nyc_parameters: NYISOParameters

def get_data(time_stamp):
  date = pd.to_datetime(time_stamp)
  date_day_before = date - pd.Timedelta(days=1)

  price_data_response = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/realtime/{date.year}{date.month:02d}{date.day:02d}realtime_zone.csv").read()
  day_ahead_price_data_response = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/damlbmp/{date.year}{date.month:02d}{date.day:02d}damlbmp_zone.csv").read()
  weather_data_tesponse = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/lfweather/{date_day_before.year}{date_day_before.month:02d}{(date_day_before.day):02d}lfweather.csv").read()
  load_forecast_data_response = urllib.request.urlopen(f"https://mis.nyiso.com/public/csv/isolf/{date.year}{date.month:02d}{date.day:02d}isolf.csv").read()

  price_data = pd.read_csv(io.BytesIO(price_data_response))
  price_data["Time Stamp"] = pd.to_datetime(price_data["Time Stamp"])
  nyc_price_data = price_data[price_data["Name"] == "N.Y.C."]
  nyc_actual_price = nyc_price_data[
    (nyc_price_data["Time Stamp"].dt.year == date.year) &
    (nyc_price_data["Time Stamp"].dt.month == date.month) &
    (nyc_price_data["Time Stamp"].dt.day == date.day) &
    (nyc_price_data["Time Stamp"].dt.hour == date.hour) &
    (nyc_price_data["Time Stamp"].dt.minute == date.minute)
    ]['LBMP ($/MWHr)'].values[0]

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

  return NYISOData(
    nyc_actual_price=nyc_actual_price,
    nyc_parameters=NYISOParameters(
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
    )
  )