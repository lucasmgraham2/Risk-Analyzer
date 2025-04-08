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

def generate_explanation(latest_features, scaled_input, feature_means, feature_stds):
    #Generate an explanation for the risk level based on feature differences.
    explanation = []
    features = ['Volatility', '50-day MA', '200-day MA', 'Momentum', 'Volume MA', 'RSI', 'Beta', 'Sharpe Ratio', 'Drawdown']

    for i, feature in enumerate(features):
        raw_value = latest_features[feature]
        scaled_value = scaled_input[0][i]
        mean_value = feature_means[feature]
        std_value = feature_stds[feature]

        if abs(scaled_value) > 1:  # Significant deviation
            if scaled_value > 0:
                explanation.append(f"{feature} is significantly higher than average (Value: {raw_value:.4f}, Mean: {mean_value:.4f}, Std: {std_value:.4f}).")
            else:
                explanation.append(f"{feature} is significantly lower than average (Value: {raw_value:.4f}, Mean: {mean_value:.4f}, Std: {std_value:.4f}).")

    return explanation

# Fetch live stock data for the Streamlit app
def fetch_live_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")
        if df.empty:
            st.error(f"Stock data for {ticker} not found. Please check the ticker symbol.")
            return None

        # Calculate features
        df["Daily Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Daily Return"].rolling(window=20).std()
        df["50-day MA"] = df["Close"].rolling(window=50).mean()
        df["200-day MA"] = df["Close"].rolling(window=200).mean()
        df["Momentum"] = df["Close"].pct_change(periods=20)
        df["Volume MA"] = df["Volume"].rolling(window=20).mean()
        df["RSI"] = 100 - (100 / (1 + df["Daily Return"].rolling(window=14).mean() / df["Daily Return"].rolling(window=14).std()))
        market_return = df["Daily Return"].rolling(window=60).mean()
        df["Beta"] = df["Daily Return"].rolling(window=60).cov(market_return) / df["Daily Return"].rolling(window=60).var()
        df["Sharpe Ratio"] = df["Daily Return"].mean() / df["Daily Return"].std()
        df["Drawdown"] = (df["Close"] / df["Close"].cummax()) - 1

        df.dropna(inplace=True)  # Drop rows with missing data
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Streamlit UI
st.title("Machine Learning Investment Risk Analyzer")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):").strip().upper()

if ticker:
    # Time period options for 1y, 3y, 5y
    time_period = st.selectbox("Select Time Period", ["1y", "3y", "5y"])
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
                'RSI': df["RSI"].iloc[-1],
                'Beta': df["Beta"].iloc[-1],
                'Sharpe Ratio': df["Sharpe Ratio"].iloc[-1],
                'Drawdown': df["Drawdown"].iloc[-1]
            }

            latest_df = pd.DataFrame([latest_features])

            # Ensure model and scaler are loaded before prediction
            if model and scaler:
                scaled_input = scaler.transform(latest_df)
                risk_level = model.predict(scaled_input)[0]  # Use the model to predict the risk level
                
                feature_means = {feature: scaler.mean_[i] for i, feature in enumerate(latest_features.keys())}
                feature_stds = {feature: scaler.scale_[i] for i, feature in enumerate(latest_features.keys())}
                explanation = generate_explanation(latest_features, scaled_input, feature_means, feature_stds)
                risk_explanations = {
                    0: "Low volatility indicates a more stable stock with lower price fluctuations.",
                    1: "Medium volatility represents a moderate risk with some price fluctuations.",
                    2: "High volatility signifies a riskier stock with significant price fluctuations."
                }
                
                # print(f"Ticker: {ticker}, Features: {latest_features}, Scaled Input: {scaled_input}, Risk Level: {risk_level}")
                if risk_level == 0:
                    risk_name = "Low"
                elif risk_level == 1:
                    risk_name = "Medium"
                else:
                    risk_name = "High"
                st.subheader(f"📈 Risk Level for {ticker}: **{risk_name}**")
                
                risk_explanation = risk_explanations.get(risk_level, "Unknown risk level")
                st.write(risk_explanation)
                
                # Chart for stock price
                st.line_chart(df["Close"], use_container_width=True)
                st.subheader("Explanation for Risk Level:")
                for line in explanation:
                    st.write(f"- {line}")
                
                # Display stock data summary
                st.subheader(f"Stock Data for {ticker} - Last 5 Days")
                st.write(df.tail())
                
                # Additional metrics
                st.subheader(f"Volatility: {latest_features['Volatility']:.4f}")
                st.line_chart(df["Volatility"], use_container_width=True)
                st.subheader(f"50-day MA: {latest_features['50-day MA']:.4f}")
                st.line_chart(df["50-day MA"], use_container_width=True)
                st.subheader(f"200-day MA: {latest_features['200-day MA']:.4f}")
                st.line_chart(df["200-day MA"], use_container_width=True)
                st.subheader(f"Momentum: {latest_features['Momentum']:.4f}")
                st.line_chart(df["Momentum"], use_container_width=True)
                st.subheader(f"Volume MA: {latest_features['Volume MA']:.4f}")
                st.line_chart(df["Volume MA"], use_container_width=True)
                st.subheader(f"RSI: {latest_features['RSI']:.4f}")
                st.line_chart(df["RSI"], use_container_width=True)
                st.subheader(f"Beta: {latest_features['Beta']:.4f}")
                st.line_chart(df["Beta"], use_container_width=True)
                st.subheader(f"Sharpe Ratio: {latest_features['Sharpe Ratio']:.4f}")
                st.line_chart(df["Sharpe Ratio"], use_container_width=True)
                st.subheader(f"Drawdown: {latest_features['Drawdown']:.4f}")
                st.line_chart(df["Drawdown"], use_container_width=True)