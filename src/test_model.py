import json
import os
import pandas as pd
import tensorflow as tf


def main():
    dirname = os.path.dirname(__file__)
    model = tf.keras.models.load_model(os.path.join(dirname, 'model/sunburst_ml.keras'))

    # date: 2025/02/13 16:45
    results = model.predict(pd.DataFrame([[2025, 2, 13, (16 * 60) + 45, 48, 37, 44, 31, 76.48]]))
    print(json.dumps({'response': results.tolist()}))

    # day ahead price: 76.48
    # actual price: 71.52

if __name__=="__main__":
    main()