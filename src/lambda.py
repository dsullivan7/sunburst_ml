import base64
import json
import os

import pandas as pd
from pickle import load

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "rb") as f:
  model = load(f)

def handler(event, context):
    results = []
    if (event and event is not None):
      body = event['body']
      if (event['isBase64Encoded']):
        body = json.loads(base64.b64decode(body))
      results = model.predict(pd.DataFrame([[
        body['year'],
        body['month'],
        body['day'],
        body['minutes'],
        body['max_temp'],
        body['min_temp'],
        body['max_wet_bulb'],
        body['min_wet_bulb'],
        body['day_ahead_price'],
        body['load_forecast']
      ]], columns=["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast"]))
      # results = pd.Series([
      #   body['year'],
      #   body['month'],
      #   body['day'],
      #   body['minutes'],
      #   body['max_temp'],
      #   body['min_temp'],
      #   body['max_wet_bulb'],
      #   body['min_wet_bulb'],
      #   body['day_ahead_price'],
      #   body['load_forecast']
      # ])

    return {
        'statusCode': 200,
        'body': json.dumps({'response': results.tolist() })
    }