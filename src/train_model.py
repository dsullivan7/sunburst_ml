import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pickle import dump
from hyperopt import tpe,STATUS_OK,Trials,fmin,hp
from hyperopt.pyll.base import scope

# best:{'max_depth': np.int64(3), 'min_samples_leaf': np.int64(8), 'min_samples_split': np.int64(18), 'n_estimators': np.int64(217)}

train_parameters = ["Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp", "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast", "Day Of Week", "Is Holiday"]
test_parameter = "LBMP ($/MWHr)"

def main():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data/data.csv'), header = 0)
    data["Time Stamp"] = pd.to_datetime(data['Time Stamp'])
    data = data.sample(frac=1)

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

    clf=RandomForestRegressor(max_depth=3, min_samples_leaf=8, min_samples_split=18, n_estimators=217)
    clf.fit(data[train_parameters], data[test_parameter])

    # def hyperparameter_tuning(params):
    #     acc=clf.score(X_test, y_test)
    #     return acc

    # best=fmin(fn=hyperparameter_tuning,
    #         space=space,
    #         algo=tpe.suggest,max_evals=10, trials=trials
    #         )

    # print("best:{}".format(best))
    with open(os.path.join(dirname, 'model/sunburst_ml.pkl'), "wb") as f:
      dump(clf, f, protocol=5)

if __name__=="__main__":
    main()