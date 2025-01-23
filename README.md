# Microsoft Stock Price Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)

This project predicts Microsoft (MSFT) stock prices using machine learning models like LSTM and XGBoost. The model is deployed as a REST API using Flask.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to predict Microsoft stock prices using historical data. Two models are implemented:
1. **LSTM (Long Short-Term Memory)**: A deep learning model for time-series data.
2. **XGBoost**: A gradient boosting model for regression tasks.

The models are trained on historical stock data from 2010 to 2023, collected using the `yfinance` library.

## Features
- Data collection using `yfinance`.
- Data preprocessing and normalization.
- Model training and evaluation.
- Deployment as a REST API using Flask.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/msft-stock-prediction.git
   cd msft-stock-prediction