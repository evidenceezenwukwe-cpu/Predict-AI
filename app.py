import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from tensorflow.keras.models import load_model

# === Load your model and scaler ===
@st.cache_resource
def load_assets():
    model = load_model("eurusd_predictor(1).keras", compile=False)
    with open("eurusd_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# === API setup ===
API_KEY = "af02217fcf4548c4a71da01704e10f68"   # Replace with your Twelve Data key
SYMBOL = "EUR/USD"

# --- Helper: fetch OHLCV ---
def fetch_timeseries(interval="1h", outputsize=30):
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    r = requests.get(url).json()
    df = pd.DataFrame(r["values"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]]
    return df[::-1].reset_index(drop=True)

# --- Helper: fetch indicators ---
def fetch_indicator(indicator, interval="1day", outputsize=30, extra_params=""):
    url = f"https://api.twelvedata.com/{indicator}?symbol={SYMBOL}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}{extra_params}"
    r = requests.get(url).json()
    df = pd.DataFrame(r["values"])
    for col in df.columns:
        if col != "datetime":
            df[col] = df[col].astype(float)
    df = df.drop(columns=["datetime"])
    return df[::-1].reset_index(drop=True)

# --- Main prediction function ---
def predict():
    ohlcv = fetch_timeseries("1h", 30)
    rsi   = fetch_indicator("rsi", interval="1day")[["rsi"]]
    ema   = fetch_indicator("ema", interval="1day")[["ema"]]
    sma   = fetch_indicator("sma", interval="1day")[["sma"]]

    macd_raw = fetch_indicator("macd", interval="1day", extra_params="&series_type=close")
    if "macd" in macd_raw.columns and "signal" in macd_raw.columns:
        macd = macd_raw[["macd", "signal"]]
    elif "macd" in macd_raw.columns:
        macd = pd.DataFrame({
            "macd": macd_raw["macd"],
            "signal": 0.0
        })
    else:
        st.error("MACD data unavailable")
        return None

    # Combine features
    df = pd.concat([ohlcv, rsi, ema, sma, macd], axis=1)
    df = df.tail(30)  # keep 30 rows

    # Scale and reshape
    X = scaler.transform(df.values)
    X = np.expand_dims(X, axis=0)  # shape: (1, 30, 10)

    # Predict
    pred = model.predict(X)
    return float(pred[0][0])

# === Streamlit UI ===
st.title("📈 EUR/USD AI Predictor (LSTM)")
st.write("Click below to fetch the latest EUR/USD market data and run prediction.")

if st.button("Run Prediction"):
    try:
        prediction = predict()
        if prediction is not None:
            st.success(f"Predicted next EUR/USD movement: {prediction:.5f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
