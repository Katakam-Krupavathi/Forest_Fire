import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv'
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        print('Downloading Algerian Forest Fire dataset from UCI...')
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    
    print('Loading and cleaning dataset...')
    df = pd.read_csv(DATASET_PATH, header=1)
    df.loc[:122, 'Region'] = 0
    df.loc[122:, 'Region'] = 1
    df['Region'] = df['Region'].astype(int)
    df = df.dropna().reset_index(drop=True)
    df = df.drop(122).reset_index(drop=True)
    df.columns = df.columns.str.strip()
    for col in ['day', 'month', 'year', 'Temperature', 'RH', 'Ws']:
        df[col] = df[col].astype(int)
    for col in ['Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']:
        df[col] = df[col].astype(float)
    df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1)
    features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
    x = df[features]
    y = df['FWI']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    ridge = Ridge()
    ridge.fit(x_train_scaled, y_train)
    y_pred = ridge.predict(x_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}, MAE: {mae:.4f}")
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'ridge.pkl'), 'wb') as f:
        pickle.dump(ridge, f)
    print('Training and pickling completed.')

if __name__ == '__main__':
    train()