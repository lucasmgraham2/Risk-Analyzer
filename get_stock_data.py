import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, period="1y", interval="1d"):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        print(f"No data found for {ticker}. Check if the symbol is correct.")
        return None

    df.reset_index(inplace=True)

    # Create 'data/' folder if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = f"data/{ticker}.csv"
    df.to_csv(file_path, index=False)
    print(f"Data for {ticker} saved at {file_path}")

    return df

if __name__ == "__main__":
    ticker = input("Enter stock symbol (e.g., AAPL, TSLA, MSFT): ").upper()
    fetch_stock_data(ticker)
