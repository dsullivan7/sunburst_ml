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
      nyc_temp = weather_data[:][(weather_data["Station ID"] == "NYC") & (pd.to_datetime(weather_data["Vintage Date"]) == (pd.to_datetime(weather_data["Forecast Date"]) + pd.Timedelta(days=1)))]
      nyc_temps.append(nyc_temp)

    final_nyc_temps = pd.concat(nyc_temps, ignore_index=True)
    final_nyc_temps.set_index('Vintage Date', inplace=True)

    # import price data
    nyc_prices = []
    price_directory = './data/prices'
    for filename in os.listdir(price_directory):
      price_data = pd.read_csv(os.path.join(price_directory, filename), header = 0)
      nyc_price = price_data[:][(price_data["Name"] == "N.Y.C.")]
      nyc_prices.append(nyc_price)

    final_nyc_prices = pd.concat(nyc_prices, ignore_index=True)
    final_nyc_prices["Time Stamp"] = pd.to_datetime(final_nyc_prices['Time Stamp'])

    # import day ahead price data
    nyc_da_prices = []
    da_price_directory = './data/day_ahead_prices'
    for filename in os.listdir(da_price_directory):
      da_price_data = pd.read_csv(os.path.join(da_price_directory, filename), header = 0)
      nyc_da_price = da_price_data[:][(da_price_data["Name"] == "N.Y.C.")]
      nyc_da_prices.append(nyc_da_price)

    final_nyc_da_prices = pd.concat(nyc_da_prices, ignore_index=True)
    final_nyc_da_prices.set_index(["Time Stamp"], inplace=True)
    final_nyc_da_prices = final_nyc_da_prices.groupby(level=0).first()

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
    final_nyc_prices["Day Ahead Price"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_da_prices.loc[x.strftime('%m/%d/%Y %H:00')]["LBMP ($/MWHr)"] )

    final_nyc_prices.to_csv("./data/data.csv")

def run():
    data = pd.read_csv(os.path.join('./data/data.csv'), header = 0)

    print(data)
    print("training model")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(9,)),        # Input layer
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
        tf.keras.layers.Dense(1)                  # Output layer with a single neuron (for regression)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    X_train, X_test, y_train, y_test = train_test_split(data[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price"]], data["LBMP ($/MWHr)"], test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # model = RandomForestRegressor(n_estimators=100)
    # model.fit(data[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price"]], data["LBMP ($/MWHr)"])

    # date: 2025/02/13 16:45
    results = model.predict(pd.DataFrame([[2025, 2, 13, (16 * 60) + 45, 48, 37, 44, 31, 76.48]]))
    print("results")
    print(results)

    # day ahead price: 76.48
    # actual price: 71.52

run()