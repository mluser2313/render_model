import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_length=24, prediction_length=48):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.output_layer = nn.Linear(seq_length * d_model, prediction_length)

    def forward(self, x):
        x = self.input_layer(x) + self.positional_encoding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2).reshape(x.size(1), -1)
        x = self.output_layer(x)
        return x

# Model parameters
input_dim = 16
d_model = 64
nhead = 4
num_layers = 2

# Global placeholders for lazy loading
model = None
scaler_features = None
scaler_target = None

def preprocess_data(data):
    df = pd.DataFrame(data)
    df['Time'] = pd.to_datetime(df['Time'].str.strip())
    df = df.sort_values(by='Time')
    df['hour'] = df['Time'].dt.hour
    df['aqi'] = df['aqi'].replace(0, np.nan)
    df['aqi'] = df['aqi'].interpolate(method='linear', limit_direction='both')
    df['Clouds %_lag1'] = df['Clouds %'].shift(1)
    df['Humidity_lag1'] = df['Humidity'].shift(1)
    df['Rain_lag1'] = df['Rain'].shift(1)
    df['Temperature_lag1'] = df['Temperature'].shift(1)
    df['Wind Speed_lag1'] = df['Wind Speed'].shift(1)
    df['aqi_lag1'] = df['aqi'].shift(1)
    df['aqi_momentum'] = df['aqi'] - df['aqi_lag1']
    df['Rain_event'] = (df['Rain'] > 0).astype(int)
    df['AQI_2hr_before'] = df['aqi'].shift(2)
    df['AQI_2hr_after'] = df['aqi'].shift(-2)
    df['Rain_aqi_reduction'] = df.apply(
        lambda row: row['AQI_2hr_before'] - row['AQI_2hr_after']
        if row['Rain_event'] == 1 else np.nan,
        axis=1
    )
    df['Hours_since_rain'] = df['Rain_event'].cumsum()
    df['Hours_since_rain'] = df.groupby('Hours_since_rain').cumcount()
    df['Rain_aqi_reduction'] = df['Rain_aqi_reduction'].fillna(0)
    df = df.fillna(0)
    features = [
        'Clouds %', 'Humidity', 'Rain', 'Temperature', 'Wind Speed',
        'Clouds %_lag1', 'Humidity_lag1', 'Rain_lag1',
        'Temperature_lag1', 'Wind Speed_lag1', 'aqi_lag1', 'hour',
        'aqi_momentum', 'Rain_event', 'Rain_aqi_reduction', 'Hours_since_rain'
    ]
    return df[features]

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler_features, scaler_target
    if model is None:
        model = TransformerModel(input_dim, d_model, nhead, num_layers)
        model.load_state_dict(torch.load("aqi_transformer_model.pkl", map_location="cpu"))
        model.eval()
        scaler_features = joblib.load("scaler_features.pkl")
        scaler_target = joblib.load("scaler_target.pkl")

    clouds = list(map(float, request.form['clouds'].split(',')))
    humidity = list(map(float, request.form['humidity'].split(',')))
    rain = list(map(float, request.form['rain'].split(',')))
    temperature = list(map(float, request.form['temperature'].split(',')))
    time = request.form['time'].split(',')
    wind_speed = list(map(float, request.form['wind_speed'].split(',')))
    aqi = list(map(float, request.form['aqi'].split(',')))

    data = {
        'Clouds %': clouds,
        'Humidity': humidity,
        'Rain': rain,
        'Temperature': temperature,
        'Time': time,
        'Wind Speed': wind_speed,
        'aqi': aqi
    }

    preprocessed_data = preprocess_data(data)
    sequence_scaled = scaler_features.transform(preprocessed_data)
    features_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction_scaled = model(features_tensor).numpy()

    prediction_actual = scaler_target.inverse_transform(prediction_scaled)
    prediction_text = "Predicted AQI for the next 48 hours: " + ", ".join([str(val) for val in prediction_actual.flatten()])

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
