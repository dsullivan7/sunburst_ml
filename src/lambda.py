import json

import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('/var/sunburst_ml.keras')

def handler(event, context):
    results = model.predict(pd.DataFrame([[
        event.payload['year'],
        event.payload['month'],
        event.payload['day'],
        event.payload['minutes'],
        event.payload['max_temp'],
        event.payload['min_temp'],
        event.payload['max_wet_bulb'],
        event.payload['min_wet_bulb'],
        event.payload['day_ahead_price']
    ]]))

    return {
        'statusCode': 200,
        'body': json.dumps({'response': results.tolist() })
    }