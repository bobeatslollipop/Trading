import yfinance as yf
import pandas as pd
import numpy as np
import gurobipy as GRB


def process(data: pd.DataFrame):
    data.index = pd.to_datetime(data.index)
    data = data.fillna(method='ffill')
    return data


def fetch(ticker='VGLT', start_date = "2005-01-01", end_date = "2024-12-31"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Process the data
    daily_avg = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    daily_avg = pd.DataFrame(daily_avg, columns=[ticker])  # Ensure it's a DataFrame for consistency

    # Save to CSV
    daily_avg.to_csv(f'data/{ticker}.csv', index=True)

    print(f"Data successfully saved to {ticker}.csv")


# IR = pd.read_csv('data/DTB3.csv', index_col=0)
# IR = process(IR)
# IR.to_csv(f'data/DTB3.csv', index=True)

# ticker_list = ['SPY', 'QQQ', 'VGLT', 'GLD', 'XLU', 'IYR', 'LQD', 'IYH', 'SVXY', 'VXX', 'JAAA']
# for ticker in ticker_list:
#     try:
#         fetch(ticker)
#     except Exception as e:
#         print(f"Failed to fetch data for {ticker}: {e}")

# fetch('UEC')