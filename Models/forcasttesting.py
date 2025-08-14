import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def run_forecast(df, product, sequence_length=10, plot=True):
    # Filter for the selected product
    product_df = df[df['Product'] == product]
    daily_sales = product_df.groupby('Date')['Units_Sold'].sum().reset_index().sort_values('Date')
    if len(daily_sales) <= sequence_length:
        raise ValueError("Not enough data for this product.")

    # Scale units sold
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_units = scaler.fit_transform(daily_sales[['Units_Sold']])
    X, y = create_sequences(scaled_units.flatten(), sequence_length=sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # For LSTM input

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, batch_size=1, epochs=25, validation_data=(X_test, y_test), verbose=0)

    # Optionally plot training history
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Forecast 7 days ahead
    last_sequence = X[-1]
    forecast = []
    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
        forecast.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred).reshape(sequence_length, 1)
    forecast_units = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Dates for forecast
    last_date = pd.to_datetime(daily_sales['Date'].iloc[-1])
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

    # Optionally plot forecast
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(daily_sales['Date'][-30:], daily_sales['Units_Sold'][-30:], label='Actual Sales', marker='o')
        plt.plot(forecast_dates, forecast_units, label='Forecast (7 Days)', marker='x', linestyle='--', color='orange')
        plt.title(f"{product} Sales Forecast - LSTM")
        plt.xlabel("Date")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return forecast_dates, forecast_units

# Example usage (for testing only, remove in production):
if __name__ == "__main__":
    df = pd.read_csv("synthetic_retail_sales.csv")  # Only for local testing!
    product = df['Product'].unique()[0]
    run_forecast(df, product)