from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import xgboost as xgb

# Initialize Flask app
app = Flask(__name__)

# Load models
lstm_model = load_model("../models/msft_lstm_model.keras")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("../models/msft_xgboost_model.json")

# Load scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Preprocess input data
def preprocess_input(data):
    data = data[['Open', 'High', 'Low', 'Volume']]
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# LSTM prediction endpoint
@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    input_data = request.json
    df = pd.DataFrame(input_data)
    processed_data = preprocess_input(df)
    processed_data = processed_data.reshape((processed_data.shape[0], 1, processed_data.shape[1]))
    predictions = lstm_model.predict(processed_data)
    predictions_actual = scaler.inverse_transform(np.concatenate((processed_data.reshape(processed_data.shape[0], -1), predictions.reshape(-1, 1)), axis=1))[:, -1]
    return jsonify(predictions_actual.tolist())

# XGBoost prediction endpoint
@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    input_data = request.json
    df = pd.DataFrame(input_data)
    processed_data = preprocess_input(df)
    predictions = xgb_model.predict(processed_data)
    predictions_actual = scaler.inverse_transform(np.concatenate((processed_data, predictions.reshape(-1, 1)), axis=1))[:, -1]
    return jsonify(predictions_actual.tolist())

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)