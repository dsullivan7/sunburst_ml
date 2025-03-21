import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pickle import dump
from hyperopt import tpe,STATUS_OK,Trials,fmin,hp
from hyperopt.pyll.base import scope
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# best:{'max_depth': np.int64(3), 'min_samples_leaf': np.int64(8), 'min_samples_split': np.int64(18), 'n_estimators': np.int64(217)}

train_parameters = ["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast", "Day Of Week", "Is Holiday"]
test_parameter = "LBMP ($/MWHr)"

def create_sequences(data, target, n_steps, step_interval):
    X, y = [], []
    for i in range(len(data) - ((n_steps - 1) * step_interval)):
        X.append(data[i:i + (n_steps * step_interval): step_interval])
        y.append(target[i+((n_steps - 1) * step_interval)])
    return np.array(X), np.array(y)

def test():
   X, y = [
      [0] * 5,
      [1] * 5,
      [2] * 5,
      [3] * 5,
      [4] * 5,
      [5] * 5,
      [6] * 5,
      [7] * 5,
      [8] * 5,
      [9] * 5,
   ], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

   res = create_sequences(X, y, 4, 3)
   print(res)

def main():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data/data.csv'), header = 0)
    data["Time Stamp"] = pd.to_datetime(data['Time Stamp'])
    data.sort_values("LBMP ($/MWHr)", inplace=True, ascending=False)
    data.reset_index(inplace=True)
    # data["Prev Day Price"] = data["LBMP ($/MWHr)"].shift(290)
    # data["Prev Week Price"] = data["LBMP ($/MWHr)"].shift(290 * 7)
    # data.dropna(inplace=True)

    # train_scaler = MinMaxScaler()
    # data[train_parameters] = train_scaler.fit_transform(data[train_parameters])

    # target_scaler = MinMaxScaler()
    # data[test_parameter] = target_scaler.fit_transform(data[[test_parameter]])

    # X_train, X_test, y_train, y_test = train_test_split(data[train_parameters], data[test_parameter], test_size=0.2, shuffle=False)

    # train_data = xgb.DMatrix(X_train, label=y_train)
    # test_data = xgb.DMatrix(X_test, label=y_test)

    # params = {
    #     "objective": "reg:squarederror",  # Regression task
    #     "eval_metric": "rmse",
    #     "learning_rate": 0.1,
    #     "max_depth": 6,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8
    # }

    # # Train the model
    # model = xgb.train(params, train_data, num_boost_round=100, evals=[(test_data, "Test")], early_stopping_rounds=10)

    # model.save_model(os.path.join(dirname, "model/sunburst_ml.json"))

    # n_steps = 7
    # X, y = create_sequences(data[train_parameters].values, data[test_parameter].values, n_steps, 290)
    # split = int(0.8 * len(X))
    # X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(n_steps, X.shape[2])),
    #   tf.keras.layers.Dropout(0.2),
    #   tf.keras.layers.LSTM(50, return_sequences=False),
    #   tf.keras.layers.Dropout(0.2),
    #   tf.keras.layers.Dense(25, activation='relu'),
    #   tf.keras.layers.Dense(1)  # Single output for price prediction
    # ])
    # model.compile(optimizer="adam", loss="mse")

    # model.fit(
    #   X_train, y_train,
    #   epochs=50,  # Increase epochs for better learning
    #   batch_size=32,
    #   validation_data=(X_test, y_test),
    #   verbose=1
    # )

    # model.save(os.path.join(dirname, './model/sunburst_ml_lstm.keras'))
    # with open(os.path.join(dirname, 'model/train_scaler_tf.pkl'), "wb") as f:
    #   dump(train_scaler, f, protocol=5)
    # with open(os.path.join(dirname, 'model/target_scaler_tf.pkl'), "wb") as f:
    #   dump(target_scaler, f, protocol=5)

    # y_pred = model.predict(X_test)

    # # Convert predictions back to original scale
    # y_pred_actual = target_scaler.inverse_transform(y_pred)
    # y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.plot(data[["Time Stamp", "LBMP ($/MWHr)"]], label="Electricity Price")
    # plt.legend()
    # plt.title("NYISO Electricity Price Prediction")
    # plt.show()

    data.plot(y="LBMP ($/MWHr)", style='o')
    plt.grid(True)
    plt.show()
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(10,)),        # Input layer
    #     tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    #     tf.keras.layers.Dense(1)                  # Output layer with a single neuron (for regression)
    # ])
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # model = RandomForestRegressor(n_estimators=100)
    # model.fit(train.iloc[::-1][["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast", "Day Of Week", "Is Holiday"]], train.iloc[::-1]["LBMP ($/MWHr)"])

    # results = model.predict(test[["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast", "Day Of Week", "Is Holiday"]])

    # clf=RandomForestRegressor(max_depth=3, min_samples_leaf=8, min_samples_split=18, n_estimators=217)
    # clf.fit(data[train_parameters], data[test_parameter])

    # def hyperparameter_tuning(params):
    #     acc=clf.score(X_test, y_test)
    #     return acc

    # best=fmin(fn=hyperparameter_tuning,
    #         space=space,
    #         algo=tpe.suggest,max_evals=10, trials=trials
    #         )

    # print("best:{}".format(best))
    # with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "wb") as f:
    #   dump(clf, f, protocol=5)

if __name__=="__main__":
    main()