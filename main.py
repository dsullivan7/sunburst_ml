import os
import pandas as pd

def run():
    # import weather data
    nyc_temps = []
    weather_directory = './data/weather'
    for filename in os.listdir(weather_directory):
      weather_data = pd.read_csv(os.path.join(weather_directory, filename), header = 0)
      nyc_temp = weather_data[:][(weather_data["Station ID"] == "NYC") & (weather_data["Vintage Date"] == weather_data["Forecast Date"])]
      nyc_temps.append(nyc_temp)

    final_nyc_temps = pd.concat(nyc_temps, ignore_index=True)

    # import price data
    nyc_prices = []
    price_directory = './data/prices'
    for filename in os.listdir(price_directory):
      price_data = pd.read_csv(os.path.join(price_directory, filename), header = 0)
      nyc_price = price_data[:][(price_data["Name"] == "N.Y.C.")]
      nyc_prices.append(nyc_price)

    final_nyc_prices = pd.concat(nyc_prices, ignore_index=True)

    final_nyc_temps.set_index('Forecast Date', inplace=True)
    final_nyc_prices["Max Temp"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.split(" ")[0]]["Max Temp"] )
    final_nyc_prices["Min Temp"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.split(" ")[0]]["Min Temp"] )
    final_nyc_prices["Max Wet Bulb"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.split(" ")[0]]["Max Wet Bulb"] )
    final_nyc_prices["Min Wet Bulb"] = final_nyc_prices["Time Stamp"].apply(lambda x: final_nyc_temps.loc[x.split(" ")[0]]["Min Wet Bulb"] )

    print(final_nyc_prices)


run()