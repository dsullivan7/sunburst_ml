import os
import pandas as pd

def run():
    price_data = pd.read_csv('./data/prices/20250211realtime_zone.csv', header = 0)
    # iterate over files in the ./data/weather folder
    nyc_temps = []
    weather_directory = './data/weather'
    for filename in os.listdir(weather_directory):
      weather_data = pd.read_csv(os.path.join(weather_directory, filename), header = 0)
      nyc_temp = weather_data[:][(weather_data["Station ID"] == "NYC") & (weather_data["Vintage Date"] == weather_data["Forecast Date"])]
      print(nyc_temp)
      nyc_temps.append(nyc_temp)

    final_nyc_temps = pd.DataFrame(nyc_temps)
    print(final_nyc_temps)


run()