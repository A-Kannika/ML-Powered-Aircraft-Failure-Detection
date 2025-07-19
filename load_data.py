import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import os

RAW_DATA_PATH = 'CMAPSSData/train_FD001.txt'
OUTPUT_PATH = 'CMAPSSData/train_FD001_cleaned.csv'
NUM_SENSOR_COLS = 21

def load_data(path):
    # Define column names
    cols = ['engine_id', 'cycle', 'op_set1', 'op_set2', 'op_set3'] + [f'sensor{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep=' ', header=None)
    df.drop(columns=[26, 27], inplace=True)
    df.columns = cols
    return df

def compute_rul(df):
    # Get max cycle per engine
    max_cycle_df = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle_df.columns = ['engine_id', 'max_cycle']
    # Merge and compute RUL
    df = df.merge(max_cycle_df, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    df['label'] = df['RUL'].apply(lambda x: 1 if x <= 30 else 0)
    return df

def normalize_features(df):
    feature_cols = ['cycle', 'op_set1', 'op_set2', 'op_set3'] + [f'sensor{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(df)
    return df

def process_and_save():
    df = load_data(RAW_DATA_PATH)
    # print(df)
    df = compute_rul(df)
    # print(df)
    df = normalize_features(df)
    # print(df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Complete!!")

if __name__ == "__main__":
    process_and_save()
