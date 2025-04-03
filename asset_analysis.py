import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as GRB

##################################################
# Understanding data. 
##################################################

def fetch_ticker(ticker: str, start_date = "2005-01-01", end_date = datetime.today().strftime('%Y-%m-%d')):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily_Average'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    data = data[['Daily_Average']]
    data = data.iloc[2:]
    data.index = pd.to_datetime(data.index)
    return data

def load_ticker(ticker: str):
    return pd.read_csv(f'data/{ticker}.csv', index_col=0, parse_dates=True)
"""
Given dataframe with shape (num_days, 1), calculate the daily changes (np.ndarray), mean daily change, and std of daily changes. 
Output: 3-tuple (daily_changes, daily_changes_mean, daily_changes_std). 
"""
def calculate_daily_changes(data: pd.DataFrame, risk_adjusted=True):
    # Risk-free rate
    if risk_adjusted:
        IR = pd.read_csv('data/DTB3.csv', index_col=0, parse_dates=True).sort_index()
        IR = IR.ffill()  # Fill forward for non-trading days (weekends/holidays)
        IR = IR / 365  # Convert to daily risk-free rate (continuous compounding)
        IR = IR.loc[data.index[0]: data.index[-1]+pd.Timedelta(days=1)]

    data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Asset data must have a DateTime index.")
    

    # Daily log returns before risk-free rate. 
    daily_changes = []
    for i in range(1, len(data)):
        price_today = data.iat[i, 0]
        price_yesterday = data.iat[i - 1, 0]
        log_return = np.log(price_today / price_yesterday) * 100 # log of daily multiplicative change
        daily_changes.append(log_return)

        if risk_adjusted:
            date_prev = data.index[i-1]
            date_curr = data.index[i]
            # Calendar days between t-1 and t
            num_days = (date_curr - date_prev).days
            if num_days < 1:
                continue  # skip same-day data if any

            # Get the RF rates for the interval [t-1, t)
            rf_period = IR.loc[date_prev:date_curr]
            if rf_period.empty:
                rf_return = 0
            else:
                rf_return = rf_period.sum().iat[0]
            daily_changes[-1] -= rf_return

    daily_changes = np.array(daily_changes)
    daily_changes_mean = daily_changes.mean()
    daily_changes_std = daily_changes.std()

    return daily_changes, daily_changes_mean, daily_changes_std


def visualize_ticker(data: pd.DataFrame, ticker: str, risk_adjusted=True, start_date='2005-01-01', end_date='2025-03-27'):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = data.loc[start_date:end_date+pd.Timedelta(days=1)]
    daily_changes, daily_changes_mean, daily_changes_std = calculate_daily_changes(data, 
                                                                                   risk_adjusted=risk_adjusted)
    print("Mean daily return of {}: {}; Std: {}; Sharpe: {}."
        .format(ticker, daily_changes_mean, daily_changes_std, np.sqrt(252) * daily_changes_mean / daily_changes_std))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- Subplot 1: Time series of daily returns
    axes[0].plot(data.index[1:], data.iloc[1:, 0])
    axes[0].set_title('Asset Value Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Value')
    axes[0].set_yscale('log')
    axes[0].grid(True)

    # --- Subplot 2: Histogram of daily returns
    axes[1].hist(daily_changes, bins=150, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of Daily Returns')
    axes[1].set_xlabel('Daily Return')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)

    # Final layout
    plt.suptitle("Daily Return Analysis for {}. Mean: {:.4f}%; Std: {:.4f}%; Sharpe:{:.4f}.".format(
        ticker, daily_changes_mean, daily_changes_std, np.sqrt(252) * daily_changes_mean / daily_changes_std)
    , fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_correlation(ticker1='^GSPC', ticker2='VGLT', start_date='2010-01-01', end_date='2025-03-27'):
    data1 = fetch_ticker(ticker1)
    data2 = fetch_ticker(ticker2)
    data1 = data1.loc[data2.index[0]:]
    data2 = data2.loc[data1.index[0]:]

    ticker1_daily_change, ticker1_daily_change_mean, ticker1_daily_change_std = calculate_daily_changes(data1)
    ticker2_daily_change, ticker2_daily_change_mean, ticker2_daily_change_std = calculate_daily_changes(data2)

    corr_matrix = np.corrcoef(ticker1_daily_change, ticker2_daily_change)
    correlation = corr_matrix[0, 1]
    print("Correlation between {} and {} daily returns:".format(ticker1, ticker2), correlation)

    plt.figure(figsize=(8, 6))
    plt.scatter(ticker1_daily_change, ticker2_daily_change, alpha=0.5)
    plt.xlabel(f'{ticker1} Daily Log Multiplicative Return %')
    plt.ylabel(f'{ticker2} Daily Log Multiplicative Return %')
    plt.title(f'{ticker1} vs {ticker2} Daily Returns. Corr={correlation}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


visualize_correlation('SPY', 'UBER')
# ticker = 'VWO'
# visualize_ticker(fetch_ticker(ticker), ticker, risk_adjusted=False)


