import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load stock data from CSV file."""
    df = pd.read_csv(file_path)
    df["Close"] = df["Close"].ffill()  # Forward fill missing values
    df["Daily Return"] = df["Close"].pct_change()  # Calculate daily returns
    df["Volatility"] = df["Daily Return"].rolling(window=20).std()  # 20-day rolling volatility
    df['50-day MA'] = df['Close'].rolling(window=50).mean()  # 50-day moving average
    df['200-day MA'] = df['Close'].rolling(window=200).mean()  # 200-day moving average
    df['Momentum'] = df['Close'].pct_change(20)  # 20-day momentum (percentage change)
    df['Volume MA'] = df['Volume'].rolling(window=20).mean()  # 20-day moving average of volume
    df['RSI'] = compute_rsi(df['Close'], 14)  # 14-day RSI (Relative Strength Index)
    
    df.dropna(inplace=True)
    return df

def compute_rsi(prices, period=14):
    """Compute the Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def classify_risk(df):
    """Classify risk using a combination of features (volatility, momentum, moving averages, RSI, etc.)."""
    
    # Combine features into a single value by normalizing them
    features = ['Volatility', '50-day MA', '200-day MA', 'Momentum', 'Volume MA', 'RSI']
    
    # Normalize each feature
    for feature in features:
        df[feature + '_scaled'] = (df[feature] - df[feature].mean()) / df[feature].std()
    
    # Calculate the combined risk score by averaging the scaled features
    df['Risk Score'] = df[[feature + '_scaled' for feature in features]].mean(axis=1)
    
    # Classify risk levels based on the combined risk score
    df['Risk Level'] = np.select(
        [df['Risk Score'] <= df['Risk Score'].quantile(0.33), 
         (df['Risk Score'] > df['Risk Score'].quantile(0.33)) & (df['Risk Score'] <= df['Risk Score'].quantile(0.66)),
         df['Risk Score'] > df['Risk Score'].quantile(0.66)],
        ["Low", "Medium", "High"], 
        default="Medium"
    )
    return df

def train_model(df):
    """Train a RandomForest model to predict risk levels based on multiple features."""
    features = ['Volatility', '50-day MA', '200-day MA', 'Momentum', 'Volume MA', 'RSI']
    X = df[features]
    y = df['Risk Level']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
    
    return model, scaler

def process_all_data(data_folder):
    """Process all CSV files in the data folder and combine them."""
    all_data = []

    # Loop through each CSV file in the data folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_folder, file_name)
            print(f"Processing {file_name}...")
            df = load_data(file_path)
            df = classify_risk(df)  # Classify risk for each stock
            all_data.append(df)

    # Concatenate all data into one DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

if __name__ == "__main__":
    data_folder = "data"  # Folder containing all the stock CSV files
    
    combined_data = process_all_data(data_folder)
    model, scaler = train_model(combined_data)
    
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    print("Model and scaler saved successfully.")
