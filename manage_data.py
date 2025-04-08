import os
import glob
import yfinance as yf
import pandas as pd

# This script fetches stock data from Yahoo Finance, calculates various features, and saves the data to CSV files.
def fetch_stock_data(ticker, period="1y", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"No data found for {ticker}. Skipping...")
            return None

        df.reset_index(inplace=True)

        # Calculate features
        df["Daily Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Daily Return"].rolling(window=20).std()
        df['50-day MA'] = df['Close'].rolling(window=50).mean()
        df['200-day MA'] = df['Close'].rolling(window=200).mean()
        df['Momentum'] = df['Close'].pct_change(20)
        df['Volume MA'] = df['Volume'].rolling(window=20).mean()
        df['RSI'] = 100 - (100 / (1 + df["Daily Return"].rolling(window=14).mean() / df["Daily Return"].rolling(window=14).std()))
        market_return = df['Daily Return'].rolling(window=60).mean()  # Simulate market return
        df['Beta'] = df['Daily Return'].rolling(window=60).cov(market_return) / df['Daily Return'].rolling(window=60).var()
        df['Sharpe Ratio'] = df['Daily Return'].mean() / df['Daily Return'].std()
        df['Drawdown'] = (df['Close'] / df['Close'].cummax()) - 1

        # Save to CSV
        if not os.path.exists("data"):
            os.makedirs("data")
        file_path = f"data/{ticker}.csv"
        df.to_csv(file_path, index=False)
        print(f"Data for {ticker} saved at {file_path}")

        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}. Skipping...")
        return None

# Function to add a specific stock
def add_specific_stock(ticker, period="5y", interval="1d"):
    print(f"Fetching data for {ticker}...")
    fetch_stock_data(ticker, period=period, interval=interval)
    print(f"Data for {ticker} has been added to the data folder.")

# Function to add random risk stocks
def add_random_risk_stocks(period="5y", interval="1d"):
    tickers_list = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE", "INTC",
    "ORCL", "CRM", "CSCO", "IBM", "AMD", "QCOM", "TXN", "AVGO", "MU", "PYPL",
    "BABA", "SHOP", "SQ", "ZM", "PLTR", "UBER", "LYFT", "ABNB", "ROKU", "SNOW",
    "TWLO", "DOCU", "CRWD", "OKTA", "ZS", "NET", "FSLY", "DDOG", "SPLK", "MDB",
    "TEAM", "ATVI", "EA", "TTWO", "RBLX", "U", "DIS", "V", "MA", "AXP",
    "WMT", "TGT", "COST", "HD", "LOW", "BBY", "KR", "WBA", "CVS", "UNH",
    "PFE", "MRNA", "JNJ", "GILD", "BMY", "AMGN", "LLY", "AZN", "NVS", "REGN",
    "XOM", "CVX", "BP", "COP", "OXY", "SLB", "HAL", "PSX", "VLO", "MPC",
    "BA", "LMT", "NOC", "RTX", "GD", "TXT", "HON", "GE", "MMM", "DE",
    "CAT", "CMI", "F", "GM", "TSM", "ASML", "NXPI", "ADI", "SWKS", "KLAC",
    "LRCX", "AMAT", "TER", "ENTG", "MKSI", "NVMI", "MRVL", "ON", "WDC", "STX",
    "ZBRA", "PANW", "FTNT", "CHKP", "JNPR", "ACN", "CTSH", "INFY", "WIT", "EPAM",
    "NOW", "WDAY", "SAP", "SAS", "TYL", "ADSK", "INTU", "CDNS", "SNPS", "ANSS",
    "SPOT", "TME", "SIRI", "DISCA", "VIAC", "NWS", "FOX", "NFLX", "HBO", "AMC",
    "CNK", "IMAX", "PARA", "HLT", "MAR", "WH", "IHG", "WYNN", "MGM", "CZR",
    "PENN", "BYD", "LVS", "SGMS", "DKNG", "GAN", "RSI", "MTCH", "BMBL", "IAC",
    "YELP", "TRIP", "BKNG", "EXPE", "ABNB", "UBER", "LYFT", "DASH", "GRUB", "PTON",
    "NKE", "UAA", "LULU", "COLM", "GPS", "PVH", "RL", "VFC", "TIF", "SIG",
    "KSS", "JWN", "DDS", "M", "TJX", "ROST", "BURL", "CROX", "DECK", "SHOO",
    "ADDYY", "PUMA", "SKX", "FOSL", "GOOS", "LEVI", "AEO", "URBN", "EXPR", "ZUMZ",
    "BBBY", "RH", "W", "OSTK", "ETSY", "EBAY", "AMZN", "SHOP", "MELI", "JD",
    "BABA", "PDD", "SE", "GLBE", "WISH", "CHWY", "CVNA", "VROOM", "KMX", "AN",
    "TSCO", "AZO", "AAP", "ORLY", "GPC", "LKQ", "SPB", "SNA", "DAN", "AXTA",
    "RPM", "PPG", "SHW", "VAL", "FMC", "ALB", "SQM", "LTHM", "PLL", "LAC",
    "MP", "CC", "CE", "IFF", "AVNT", "EMN", "WLK", "UNVR", "HUN", "BCPC",
    "CTVA", "BG", "ADM", "MOS", "NTR", "CF", "AGU", "DE", "AGCO", "TTC",
    "LNN", "CNHI", "CAT", "CMI", "PCAR", "NAV", "OSK", "WNC", "GBX", "TRN",
    "F", "GM", "TSLA", "RIDE", "WKHS", "FSR", "NKLA", "XPEV", "NIO", "LI",
    "BYDDF", "STLA", "TM", "HMC", "FCAU", "VLVLY", "RACE", "TTM", "MZDAY", "NSANY",
    "VWAGY", "BMWYY", "DDAIF", "PAG", "LAD", "SAH", "AN", "GPI", "ABG", "CARG",
    "CVNA", "VROOM", "KMX", "SFT", "RUSHB", "HOG", "PII", "DOOO", "BRP", "THO",
    "WGO", "LCII", "PATK", "CWH", "MPX", "MLR", "LEA", "ADNT", "DORM", "MOD",
    "TEN", "SUP", "MGA", "BWA", "AXL", "VC", "ALSN", "GT", "CTB", "NPTN",
    "CRSR", "LOGI", "STE", "ISRG", "BSX", "ZBH", "SYK", "MDT", "EW", "ABT",
    "BAX", "BDX", "RMD", "TFX", "HOLX", "VAR", "TMO", "A", "ILMN", "PKI",
    "DGX", "LH", "MTD", "BIO", "QDEL", "NEOG", "OCX", "CDNA", "EXAS", "GH",
    "NVTA", "TWST", "BLI", "TXG", "FLGT", "FATE", "SRPT", "EDIT", "CRSP", "NTLA",
    "BEAM", "BLUE", "MRNA", "BNTX", "NVAX", "PFE", "JNJ", "GILD", "REGN", "AMGN",
    "VRTX", "BIIB", "INCY", "SRNE", "ARDX", "ACAD", "NBIX", "SAGE", "ITCI", "ALKS",
    "VIR", "RETA", "TGTX", "RCUS", "MDGL", "VKTX", "IMVT", "GOSS", "XNCR", "KNSA"
    ]
    tickers_list = list(set(tickers_list))

    for ticker in tickers_list:
        fetch_stock_data(ticker, period=period, interval=interval)

# Function to clear the data folder
def clear_data_folder():
    data_folder = "data"
    if not os.path.exists(data_folder):
        print("Data folder does not exist.")
        return

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        print("No CSV files found in the data folder.")
        return

    for file in csv_files:
        os.remove(file)
        print(f"Deleted: {file}")

    print("All CSV files have been cleared from the data folder.")

# Main function to run the script
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Add a specific stock")
    print("2. Add random risk stocks")
    print("3. Clear the data folder")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        ticker = input("Enter the stock ticker (e.g., AAPL, TSLA): ").strip().upper()
        add_specific_stock(ticker)
    elif choice == "2":
        add_random_risk_stocks()
    elif choice == "3":
        clear_data_folder()
    else:
        print("Invalid choice. Please run the script again.")