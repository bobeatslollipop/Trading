import yfinance as yf
import pandas as pd
import numpy as np
import gurobipy as GRB

def fetch(ticker='VGLT', start_date = "2005-01-01", end_date = "2025-03-27"):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate daily average of OHLC prices
    data['Daily_Average'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)

    # Keep only the Daily_Average column
    daily_avg = data[['Daily_Average']]
    daily_avg = daily_avg.iloc[2:]

    # Save to CSV
    daily_avg.to_csv(f'data/{ticker}.csv', index=True)

    print(f"Data successfully saved to {ticker}.csv")


def process(data):
    data.index = pd.to_datetime(data.index)
    data = data.fillna(method='ffill')
    return data

# IR = pd.read_csv('data/DTB3.csv', index_col=0)
# IR = process(IR)
# IR.to_csv(f'data/DTB3.csv', index=True)
fetch('AMZN')