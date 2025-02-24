import json
import os
import pandas as pd
from pickle import load

def main():
    dirname = os.path.dirname(__file__)
    # model = tf.keras.models.load_model(os.path.join(dirname, 'model/sunburst_ml.pkl'))

    with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "rb") as f:
      model = load(f)

    # date: 2025/02/13 16:45
    results = model.predict(
      pd.DataFrame([[2025, 2, 13, (16 * 60) + 45, 48, 37, 44, 31, 76.48, 6133]],
      columns=["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast"])
    )
    print(json.dumps({'response': results.tolist()}))

    # day ahead price: 76.48
    # actual price: 71.52

if __name__=="__main__":
    main()