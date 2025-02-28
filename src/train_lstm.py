import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pickle import dump
import matplotlib.pyplot as plt

# Define training parameters and target variable
TRAIN_PARAMETERS = [
    "Year", "Month", "Day", "Minutes", "Max Temp", "Min Temp",
    "Max Wet Bulb", "Min Wet Bulb", "Day Ahead Price", "Load Forecast",
    "Day Of Week", "Is Holiday"
]
TARGET_PARAMETER = "LBMP ($/MWHr)"

def create_sequences(data, target, n_steps, step_interval):
    """Create sequences for LSTM input with specified step interval"""
    X, y = [], []
    for i in range(len(data) - ((n_steps - 1) * step_interval)):
        X.append(data[i:i + (n_steps * step_interval):step_interval])
        y.append(target[i + ((n_steps - 1) * step_interval)])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build and return LSTM model architecture"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    return model

def main():
    # Set paths
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, 'data/data.csv')
    model_path = os.path.join(dirname, 'model/lstm_price_predictor.keras')
    train_scaler_path = os.path.join(dirname, 'model/train_scaler_lstm.pkl')
    target_scaler_path = os.path.join(dirname, 'model/target_scaler_lstm.pkl')

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = pd.read_csv(data_path)
    data["Time Stamp"] = pd.to_datetime(data['Time Stamp'])
    data.sort_values("Time Stamp", inplace=True)

    # Scale features
    train_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    data[TRAIN_PARAMETERS] = train_scaler.fit_transform(data[TRAIN_PARAMETERS])
    data[TARGET_PARAMETER] = target_scaler.fit_transform(data[[TARGET_PARAMETER]])

    # Create sequences for LSTM
    print("Creating sequences...")
    n_steps = 24  # Use last 24 time steps to predict next value
    step_interval = 12  # Use every time step
    X, y = create_sequences(
        data[TRAIN_PARAMETERS].values,
        data[TARGET_PARAMETER].values,
        n_steps,
        step_interval
    )

    # Split data
    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train model
    print("Training model...")
    model = build_lstm_model(input_shape=(n_steps, X.shape[2]))

    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.4f}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Convert predictions back to original scale
    y_pred_actual = target_scaler.inverse_transform(y_pred)
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[-100:], label='Actual Prices')
    plt.plot(y_pred_actual[-100:], label='Predicted Prices')
    plt.title('NYISO Electricity Price: Actual vs Predicted (Last 100 Points)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($/MWHr)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model and scalers
    print("\nSaving model and scalers...")
    model.save(model_path)
    with open(train_scaler_path, 'wb') as f:
        dump(train_scaler, f)
    with open(target_scaler_path, 'wb') as f:
        dump(target_scaler, f)

    print("Training complete! Model and scalers saved.")

if __name__ == "__main__":
    main()