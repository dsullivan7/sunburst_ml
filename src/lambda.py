import base64
import json

import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('/var/sunburst_ml.keras')

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
      ]]))
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