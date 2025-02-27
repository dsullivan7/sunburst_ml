import json
import os
import pandas as pd
from pickle import load
import xgboost as xgb

# {
#   year: 2025,
#   month: 2,
#   day: 26,
#   minutes: 1020,
#   max_temp: 53,
#   min_temp: 40,
#   max_wet_bulb: 43,
#   min_wet_bulb: 37,
#   day_ahead_price: 64.62,
#   load_forecast: 5809,
#   day_of_week: 2,
#   is_holiday: false
# }
# {
#   predictedPrice: 69.54478988781463,
#   actualPrice: 54.23,
#   dayAheadPrice: 64.62,
#   timeStamp: '2025-02-26T17:00:00'
# }

def main():
    dirname = os.path.dirname(__file__)
    # model = tf.keras.models.load_model(os.path.join(dirname, 'model/sunburst_ml.pkl'))

    model = xgb.Booster()
    model.load_model(os.path.join(dirname, "model/sunburst_ml.json"))

    with open(os.path.join(dirname, "model/train_scaler.pkl"), "rb") as file:
      train_scaler = load(file)
    with open(os.path.join(dirname, "model/target_scaler.pkl"), "rb") as file:
      target_scaler = load(file)

    data = [[2025, 2, 26, 1020, 53, 40, 43, 37, 64.62, 5809, 2, False, 56.53, 209.82]]
    data_scaled = pd.DataFrame(train_scaler.transform(data), columns=["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast", "Day Of Week", "Is Holiday", "Prev Day Price", "Prev Week Price"])
    dmatrix = xgb.DMatrix(data_scaled)

    # date: 2025/02/13 16:45
    results_scaled = model.predict(dmatrix)
    result = target_scaler.inverse_transform(results_scaled.reshape(-1, 1))[0][0]
    print('result')
    print(result)

    # day ahead price: 76.48
    # actual price: 71.52

if __name__=="__main__":
    main()