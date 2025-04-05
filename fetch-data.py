import yfinance as yf
import pandas as pd
import numpy as np
import gurobipy as GRB
import pandas_datareader.data as web


def process(data: pd.DataFrame):
    data.index = pd.to_datetime(data.index)
    data = data.fillna(method='ffill')
    return data


def fetch(ticker='VGLT', start_date="2005-01-01", end_date="2024-12-31"):
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

# # Fetch VIX data from FRED
# vix_data = web.DataReader('VIXCLS', 'fred', start_date, end_date)
# vix_data = pd.DataFrame(vix_data, columns=['VIX'])
# vix_data.to_csv('data/VIX.csv', index=True)
# print(vix_data)
#
# # Fetch Federal Funds Rate data from FRED
# fed_funds_data = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
# fed_funds_data = pd.DataFrame(fed_funds_data, columns=['FEDFUNDS'])
# fed_funds_data.to_csv('data/FEDFUNDS.csv', index=True)
# print(fed_funds_data)


# # Load the file, skipping the first row and specifying column names
# df = pd.read_excel("data/VIX_futures_data.xlsx", skiprows=1, names=["Date", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "days_to_expiration"])
#
# df['Date'] = pd.to_datetime(df['Date'], origin='1899-12-30', unit='D')
# df['Date'] = df['Date'].dt.normalize()
# df.set_index('Date', inplace=True)
#
# df.to_csv(f'data/VIX_futures.csv', index=True)

if __name__ == "__main__":
    start_date = '2005-01-01'
    end_date = '2019-01-01'


