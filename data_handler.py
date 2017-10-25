import numpy as np
import pandas as pd

import os.path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split



class Sample:
    X = None
    y = None
    def __init__(self, X, y = None):
        self.X = X
        self.y = y

def load_data():
    df_e10 = pd.read_csv('e10.csv')
    df_e12 = pd.read_csv('e12.csv')
    # df_e12labls = pdf.read_csv('cell.identity.csv')

    combined = pd.concat([df_e10,df_e12])
    combined = combined.drop(combined.columns[0], axis=1)

    # normalize data: use minmax or standardize?
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x = combined.values.astype(float)
    x_scaled = scaler.fit_transform(x)

    x_train, x_test = train_test_split(x_scaled, test_size=0.2, random_state=42)
    return x_train, x_test

load_data()
