import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pickle import dump


def main():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data/data.csv'), header = 0)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(10,)),        # Input layer
    #     tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    #     tf.keras.layers.Dense(1)                  # Output layer with a single neuron (for regression)
    # ])
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # X_train, X_test, y_train, y_test = train_test_split(data[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast"]], data["LBMP ($/MWHr)"], test_size=0.2, random_state=42)
    # model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    model = RandomForestRegressor(n_estimators=100)
    model.fit(data[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast"]], data["LBMP ($/MWHr)"])

    with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "wb") as f:
      dump(model, f, protocol=5)

if __name__=="__main__":
    main()