import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import tensorflow as tf

def get_data():
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

    final_nyc_prices.to_csv("./data/data.csv")

def run():
    data = pd.read_csv(os.path.join('./data/data.csv'), header = 0)

    print(data)
    print("training model")

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(1,)),        # Input layer
    #     tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    #     tf.keras.layers.Dense(1)                  # Output layer with a single neuron (for regression)
    # ])
    # model.compile(optimizer='adam', loss='mean_squared_error')
    model = RandomForestRegressor(n_estimators=100)
    model.fit(data[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb"]], data["LBMP ($/MWHr)"])

    # date: 2025/02/13 16:45
    results = model.predict([[2025, 2, 13, (16 * 60) + 45, 46, 34, 43, 31]])
    print("results")
    print(results)

    # target is 71.52


run()