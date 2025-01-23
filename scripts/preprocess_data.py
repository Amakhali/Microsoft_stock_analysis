import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data_path):
    # Load data
    data = pd.read_csv(data_path)
    
    # Select relevant features
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler