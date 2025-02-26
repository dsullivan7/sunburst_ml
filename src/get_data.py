import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from sklearn.model_selection import train_test_split

def main():
    dirname = os.path.dirname(__file__)

    # import weather data
    nyc_temps = []
    weather_directory = os.path.join(dirname, 'data/weather')
    for filename in os.listdir(weather_directory):
      weather_data = pd.read_csv(os.path.join(weather_directory, filename), header = 0)
      nyc_temp = weather_data[:][(weather_data["Station ID"] == "NYC") & (pd.to_datetime(weather_data["Vintage Date"]) == (pd.to_datetime(weather_data["Forecast Date"]) + pd.Timedelta(days=1)))]
      nyc_temps.append(nyc_temp)

    final_nyc_temps = pd.concat(nyc_temps, ignore_index=True)
    final_nyc_temps.set_index('Vintage Date', inplace=True)

    # import load forecast data
    nyc_loads = []
    load_directory = os.path.join(dirname, 'data/load')
    for filename in os.listdir(load_directory):
      load_data = pd.read_csv(os.path.join(load_directory, filename), header = 0)
      load_year = filename[0:4]
      load_month = filename[4:6]
      load_day = filename[6:8]
      nyc_load = load_data[load_data["Time Stamp"].str.slice(0, 10) == load_month + '/' + load_day + '/' + load_year][["Time Stamp", "N.Y.C."]]
      nyc_loads.append(nyc_load)

    final_nyc_loads = pd.concat(nyc_loads, ignore_index=True)
    final_nyc_loads.set_index('Time Stamp', inplace=True)
    final_nyc_loads = final_nyc_loads.groupby(level=0).first()

    # import day ahead price data
    nyc_da_prices = []
    da_price_directory = os.path.join(dirname, 'data/day_ahead_prices')
    for filename in os.listdir(da_price_directory):
      da_price_data = pd.read_csv(os.path.join(da_price_directory, filename), header = 0)
      nyc_da_price = da_price_data[:][(da_price_data["Name"] == "N.Y.C.")]
      nyc_da_prices.append(nyc_da_price)

    final_nyc_da_prices = pd.concat(nyc_da_prices, ignore_index=True)
    final_nyc_da_prices.set_index(["Time Stamp"], inplace=True)
    final_nyc_da_prices = final_nyc_da_prices.groupby(level=0).first()

    # import price data
    nyc_prices = []
    price_directory = os.path.join(dirname, 'data/prices')
    for filename in os.listdir(price_directory):
      price_data = pd.read_csv(os.path.join(price_directory, filename), header = 0)
      nyc_price = price_data[:][(price_data["Name"] == "N.Y.C.")]
      nyc_prices.append(nyc_price)

    final_nyc_prices = pd.concat(nyc_prices, ignore_index=True)
    final_nyc_prices["Time Stamp"] = pd.to_datetime(final_nyc_prices['Time Stamp'])

    cal = calendar()
    holidays = cal.holidays(start=final_nyc_prices["Time Stamp"].min(), end=final_nyc_prices["Time Stamp"].max())

    print("concatenating data")
    final_nyc_prices["Max Temp"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Max Temp"] )
    final_nyc_prices["Min Temp"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Min Temp"] )
    final_nyc_prices["Max Wet Bulb"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Max Wet Bulb"] )
    final_nyc_prices["Min Wet Bulb"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Min Wet Bulb"] )
    final_nyc_prices["Year"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.year)
    final_nyc_prices["Month"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.month)
    final_nyc_prices["Day"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.day)
    final_nyc_prices["Hour"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.hour)
    final_nyc_prices["Minutes"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.hour * 60 + x.minute)
    final_nyc_prices["Day Of Week"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.day_of_week )
    final_nyc_prices["Is Holiday"] = final_nyc_prices["Time Stamp"].isin(holidays)
    final_nyc_prices["Day Ahead Price"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_da_prices.loc[x.strftime('%m/%d/%Y %H:00')]["LBMP ($/MWHr)"] )
    final_nyc_prices["Load Forecast"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_loads.loc[x.strftime('%m/%d/%Y %H:00')]["N.Y.C."] )

    final_nyc_prices.to_csv(os.path.join(dirname, "data/data.csv"))

if __name__=="__main__":
    main()