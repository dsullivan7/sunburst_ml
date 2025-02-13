import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run():
    # import weather data
    nyc_temps = []
    weather_directory = './data/weather'
    for filename in os.listdir(weather_directory):
      weather_data = pd.read_csv(os.path.join(weather_directory, filename), header = 0)
      nyc_temp = weather_data[:][(weather_data["Station ID"] == "NYC") & (weather_data["Vintage Date"] == weather_data["Forecast Date"])]
      nyc_temps.append(nyc_temp)

    final_nyc_temps = pd.concat(nyc_temps, ignore_index=True)
    final_nyc_temps.set_index('Forecast Date', inplace=True)

    # import price data
    nyc_prices = []
    price_directory = './data/prices'
    for filename in os.listdir(price_directory):
      price_data = pd.read_csv(os.path.join(price_directory, filename), header = 0)
      nyc_price = price_data[:][(price_data["Name"] == "N.Y.C.")]
      nyc_prices.append(nyc_price)

    final_nyc_prices = pd.concat(nyc_prices, ignore_index=True)
    final_nyc_prices["Time Stamp"] = pd.to_datetime(final_nyc_prices['Time Stamp'])

    print("concatenating data")
    final_nyc_prices["Max Temp"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Max Temp"] )
    final_nyc_prices["Min Temp"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Min Temp"] )
    final_nyc_prices["Max Wet Bulb"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Max Wet Bulb"] )
    final_nyc_prices["Min Wet Bulb"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.strftime('%m/%d/%Y')]["Min Wet Bulb"] )
    final_nyc_prices["Year"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.year)
    final_nyc_prices["Month"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.month)
    final_nyc_prices["Day"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.day)
    final_nyc_prices["Minutes"] = final_nyc_prices["Time Stamp"].apply(lambda x: x.hour * 60 + x.minute)

    print(final_nyc_prices)
    print("training model")

    model = RandomForestRegressor(n_estimators=100)
    model.fit(final_nyc_prices[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb"]], final_nyc_prices["LBMP ($/MWHr)"])

    # date: 2025/02/13 16:45
    results = model.predict([[2025, 2, 13, (16 * 60) + 45, 46, 34, 43, 31]])
    print("results")
    print(results)

    # target is 71.52


run()