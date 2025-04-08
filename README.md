# 📈 Stock Risk Analyzer

The **Stock Risk Analyzer** is a machine learning-based tool that evaluates the potential risk level of stocks based on historical data, financial indicators, and predictive modeling. It helps investors make informed decisions by quantifying volatility and forecasting risk levels.

---

## 🚀 Features

- 🔍 Analyzes historical stock data and key financial metrics
- 🤖 Utilizes a trained machine learning model to predict risk levels
- 📊 Visualizes risk categories (e.g., Low, Medium, High)
- 🧠 Supports retraining with new data for improved accuracy
- 🛠️ Easy-to-run command line interface

---

## ⚙️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Risk-Analyzer.git
cd Risk-Analyzer
```

### Step 2: Set Up the Environment

#### Prerequisites
- Python 3.10 or later ([Download from python.org](https://python.org))

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Data

Choose one of the following options:

| Option | Description | Command |
|--------|-------------|---------|
| **1️⃣ Add Specific Stock** | Fetch data for an individual stock ticker | `python manage_data.py` → Select option 1 → Enter ticker (e.g., AAPL) |
| **2️⃣ Add Random Stocks** | Fetch data for a predefined list of stocks | `python manage_data.py` → Select option 2 |
| **3️⃣ Clear Data** | Delete all CSV files in the data folder | `python manage_data.py` → Select option 3 |

### Step 4: Train the Model
```bash
python train_model.py
```

This process will:
- Process all stock data in the `data` folder
- Train an XGBoost classification model
- Save the trained model (`model.pkl`) and scaler (`scaler.pkl`)

### Step 5: Launch the Application
```bash
streamlit run app.py
```

### Step 6: Use the Web Interface

1. Open the URL provided by Streamlit (typically `http://localhost:8501`)
2. Enter a stock ticker symbol (e.g., AAPL, TSLA)
3. Select a time period (1y, 3y, or 5y)
4. View the risk analysis results including:
   - Risk level classification
   - Detailed explanations
   - Key performance metrics
