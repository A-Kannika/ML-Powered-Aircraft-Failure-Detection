import pandas as pd
import numpy as np
import time
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.models import Sequential
from keras.src.layers import BatchNormalization, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

CLEANED_DATA_PATH = 'CMAPSSData/train_FD001_cleaned.csv'

def load_cleaned_data(path):
    df = pd.read_csv(path)
    return df

def generate_sequences(df, window_size=15, max_per_engine=100):
    sequences = []
    labels = []
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col not in ['engine_id', 'cycle', 'RUL']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id].reset_index(drop=True)
        total_windows = len(engine_df) - window_size
        if total_windows <= 0:
            continue
        indices = np.random.choice(total_windows, min(max_per_engine, total_windows), replace=False)
        for i in indices:
            seq = engine_df.iloc[i:i + window_size]
            label = engine_df.iloc[i + window_size]['RUL']
            sequences.append(seq[feature_cols].values)
            labels.append(label)
    X = np.array(sequences)
    y = np.array(labels)
    return X, y

def train_test_splitting(df):
    X, y = generate_sequences(df)
    y = y.reshape(-1, 1)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model_RandomForestRegressor(df):
    print("===== Random Forest =====")
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_splitting(df)
    print(f"Data generation and split took {time.time() - start:.2f} seconds")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=42)

    start = time.time()
    model.fit(X_train_flat, y_train.reshape(-1))
    print(f"Model training took {time.time() - start:.2f} seconds")

    y_pred = model.predict(X_test_flat)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R² Score: {r2:.3f}")

def build_model_XGBRegressor(df):
    print("===== XGBoost =====")
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_splitting(df)
    print(f"Data generation and split took {time.time() - start:.2f} seconds")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42)

    start = time.time()
    model.fit(X_train_flat, y_train)
    print(f"Model training took {time.time() - start:.2f} seconds")

    y_pred = model.predict(X_test_flat)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R² Score: {r2:.3f}")

def LSTM_model(input_shape):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

import matplotlib.pyplot as plt
def build_model_LSTM(df):
    print("===== LSTM =====")
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_splitting(df)
    print(f"Data generation and split took {time.time() - start:.2f} seconds")

    model = LSTM_model((X_train.shape[1], X_train.shape[2]))

    start = time.time()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=64,
              validation_split=0.1, callbacks=[early_stop], verbose=1)
    print(f"Model training took {time.time() - start:.2f} seconds")

    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R² Score: {r2:.3f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='LSTM Prediction')
    plt.title('Actual vs LSTM Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = load_cleaned_data(CLEANED_DATA_PATH)
    build_model_RandomForestRegressor(df)
    build_model_XGBRegressor(df)
    build_model_LSTM(df)

