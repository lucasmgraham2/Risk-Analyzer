import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
from train_model import classify_risk

# Load trained model & scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Ensure the files are in the correct location.")
        return None, None


model, scaler = load_model()

def fetch_live_stock_data(ticker, period="1y"):
    """Fetch real-time stock data."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")
        if df.empty:
            st.error(f"Stock data for {ticker} not found. Please check the ticker symbol.")
            return None
        df["Daily Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Daily Return"].rolling(window=20).std()
        
        # Calculate additional features for the model
        df["50-day MA"] = df["Close"].rolling(window=50).mean()
        df["200-day MA"] = df["Close"].rolling(window=200).mean()
        df["Momentum"] = df["Close"].pct_change(periods=5)  # 5-day momentum
        df["Volume MA"] = df["Volume"].rolling(window=50).mean()
        df["RSI"] = 100 - (100 / (1 + df["Daily Return"].rolling(window=14).mean() / df["Daily Return"].rolling(window=14).std()))  # 14-day RSI

        df.dropna(inplace=True)  # Drop rows with missing data
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Streamlit UI
st.title("Machine Learning Investment Risk Analyzer")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):").strip().upper()

if ticker:
    time_period = st.selectbox("Select Time Period", ["1y", "6mo", "1mo", "5d"])
    df = fetch_live_stock_data(ticker, period=time_period)
    
    if df is not None:
        # Check if there is enough data for prediction
        if len(df) < 20:  # Need at least 20 data points to calculate volatility
            st.error("Not enough data available to calculate volatility for prediction.")
        else:
            # Calculate the features used in the model
            latest_features = {
                'Volatility': df["Volatility"].iloc[-1],
                '50-day MA': df["50-day MA"].iloc[-1],
                '200-day MA': df["200-day MA"].iloc[-1],
                'Momentum': df["Momentum"].iloc[-1],
                'Volume MA': df["Volume MA"].iloc[-1],
                'RSI': df["RSI"].iloc[-1]
            }

            latest_df = pd.DataFrame([latest_features])

            # Ensure model and scaler are loaded before prediction
            if model and scaler:
                scaled_input = scaler.transform(latest_df)
                risk_level = model.predict(scaled_input)[0]

                st.subheader(f"📈 Risk Level for {ticker}: **{risk_level}**")
                st.line_chart(df["Close"], use_container_width=True)

                # Additional metrics
                st.subheader(f"📉 Volatility: {latest_features['Volatility']:.4f}")
                st.line_chart(df["Volatility"], use_container_width=True)

                # Risk level explanation
                risk_explanations = {
                    "Low": "Low volatility indicates a more stable stock with lower price fluctuations.",
                    "Medium": "Medium volatility represents a moderate risk with some price fluctuations.",
                    "High": "High volatility signifies a riskier stock with significant price fluctuations."
                }

                # Safely retrieve the explanation, defaulting to "Unknown risk level" if not found
                risk_explanation = risk_explanations.get(risk_level, "Unknown risk level")

                st.write(f"**Risk Level Explanation**: {risk_explanation}")

                # Display stock data summary
                st.subheader(f"Stock Data for {ticker} - Last 5 Days")
                st.write(df.tail())
