import os
import pandas as pd

from sklearn.model_selection import train_test_split

def main():
    dirname = os.path.dirname(__file__)

    # import weather data
    nyc_temps = []
    weather_directory = os.path(dirname, 'data/weather')
    for filename in os.listdir(weather_directory):
      weather_data = pd.read_csv(os.path.join(weather_directory, filename), header = 0)
      nyc_temp = weather_data[:][(weather_data["Station ID"] == "NYC") & (pd.to_datetime(weather_data["Vintage Date"]) == (pd.to_datetime(weather_data["Forecast Date"]) + pd.Timedelta(days=1)))]
      nyc_temps.append(nyc_temp)

    final_nyc_temps = pd.concat(nyc_temps, ignore_index=True)
    final_nyc_temps.set_index('Vintage Date', inplace=True)

    # import price data
    nyc_prices = []
    price_directory = os.path(dirname, 'data/prices')
    for filename in os.listdir(price_directory):
      price_data = pd.read_csv(os.path.join(price_directory, filename), header = 0)
      nyc_price = price_data[:][(price_data["Name"] == "N.Y.C.")]
      nyc_prices.append(nyc_price)

    final_nyc_prices = pd.concat(nyc_prices, ignore_index=True)
    final_nyc_prices["Time Stamp"] = pd.to_datetime(final_nyc_prices['Time Stamp'])

    # import day ahead price data
    nyc_da_prices = []
    da_price_directory = os.path(dirname, 'data/day_ahead_prices')
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

    final_nyc_prices.to_csv(os.path(dirname, "data/data.csv"))

if __name__=="__main__":
    main()