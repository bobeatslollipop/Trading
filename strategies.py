import yfinance as yf
import pandas as pd
import numpy as np
import gurobipy as GRB
import matplotlib.pyplot as plt
from asset_analysis import calculate_daily_changes, fetch_ticker
import math

def EWMA_crossover(data: pd.DataFrame, beta1=0.95, beta2=0.99):
    """
    Exponentially Weighted Moving Average Crossover. 
    Long position iff recent average overtakes the long-term average. 
    """
    MA1 = [data.values[0, 0]]
    MA2 = [data.values[0]]
    money = [100]  # initial investment

    for i in range(1, len(data)):
        cur_val = data.values[i, 0]
        # Update exponential moving averages
        new_MA1 = beta1 * MA1[-1] + (1 - beta1) * cur_val
        new_MA2 = beta2 * MA2[-1] + (1 - beta2) * cur_val
        MA1.append(new_MA1)
        MA2.append(new_MA2)

        # Strategy: long when MA1 > MA2, otherwise cash
        if MA1[-2] > MA2[-2]:  # yesterday’s signal
            cur_money = money[-1] * cur_val / data.values[i - 1, 0]
        else:
            cur_money = money[-1]
        money.append(cur_money)

    return np.array(money), MA1, MA2


def visualize_EWMA_crossover(ticker:str, beta1=0.95, beta2=0.99, start_date = "2005-01-01", end_date = "2025-03-27"):
    data = fetch_ticker(ticker)
    money, MA1, MA2 = EWMA_crossover(data, beta1, beta2)

    # Convert to series for easy plotting
    MA1_series = pd.Series(MA1, index=data.index)
    MA2_series = pd.Series(MA2, index=data.index)
    money_series = pd.Series(money, index=data.index)

    # Normalize data to start at 100 for better comparison
    data = data / data.iloc[0] * 100
    MA1_series = MA1_series / MA1_series.iloc[0] * 100
    MA2_series = MA2_series / MA2_series.iloc[0] * 100

    # Plot all lines on the same axis
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Asset Price', color='blue')
    plt.plot(MA1_series, label=f'MA1 (β={beta1})', color='orange', linestyle='--')
    plt.plot(MA2_series, label=f'MA2 (β={beta2})', color='green', linestyle='--')
    plt.plot(money_series, label='Cumulative Return', color='red')

    plt.title(f"EWMAC Strategy for {ticker}")
    plt.xlabel('Date')
    plt.yscale('log')
    plt.ylabel('Normalized Value (Starting at 100)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()


def SMA_crossover(data: pd.DataFrame, window=200,
        buffer_start=1, buffer_end=1, min_hold=0, max_hold=1):
    """
    Computes a simple moving average (SMA) crossover strategy.
    Args:
        data (pd.DataFrame): Price data (single column).
        fast_window (int): Window for fast SMA (e.g., 50-day).
        slow_window (int): Window for slow SMA (e.g., 200-day).
        buffer (None | float): If None, then position is 0 when below and 1 when above. 
            If a number between zero and one, then that's the portion of buffer above/below MA. 
    Returns:
        np.array: Cumulative returns (starting at 100).
        list: Fast SMA values.
        list: Slow SMA values.
    """
    # Calculate SMAs
    sma = data.rolling(window=window).mean().values.flatten()
    hold = [0]
    money = [100]  # Initial investment
    
    for i in range(1, len(data)):
        # Strategy: Long if fast_SMA > slow_SMA (yesterday's signal)
        cur_money = money[-1] * ((1-hold[-1]) * data.values[i-1, 0] 
                                 + hold[-1] * data.values[i,0]) / data.values[i-1, 0]
        money.append(cur_money)

        ratio = data.values[i,0] / sma[i]
        if buffer_start == buffer_end: # no buffer
            if min_hold == 0 and max_hold == 1: # ordinary
                hold.append(int(data.values[i,0] >= sma[i]))
            else: # with short selling:
                hold.append((data.values[i,0] >= sma[i]) * 2 - 1)
        elif math.isnan(sma[i]): # window hasn't started
            hold.append(0)
        elif ratio >= buffer_end:
            hold.append(max_hold)
        elif ratio <= buffer_start:
            hold.append(min_hold)
        elif ratio <= 1:
            hold.append((ratio - buffer_start) / (1 - buffer_start) * min_hold)
        else: # ratio > 1
            hold.append((buffer_end - ratio) / (buffer_end - 1) * max_hold)

    return np.array(money), sma, np.array(hold)


def visualize_SMA_crossover(ticker: str, window=200, 
        start_date="2005-01-01", end_date="2025-03-27", buffer_start=1, buffer_end=1, min_hold=0, max_hold=1):
    """
    Plots the SMA crossover strategy performance.
    Args:
        ticker (str): Stock symbol (e.g., "SPY").
        fast_window (int): Fast SMA window (default: 50).
        slow_window (int): Slow SMA window (default: 200).
        start_date (str): Start date for data.
        end_date (str): End date for data.
    """
    data = fetch_ticker(ticker)
    data = data.loc[start_date:end_date]
    money, sma, _ = SMA_crossover(data, window, 
        buffer_start=buffer_start, buffer_end=buffer_end, min_hold=min_hold, max_hold=max_hold)
    
    # Convert to Series for plotting
    sma_series = pd.Series(sma, index=data.index)
    money_series = pd.Series(money, index=data.index)
    
    # Normalize for comparison
    sma_series /= data.iat[window, 0] / 100  
    data /= data.iat[window, 0] / 100
    money_series /= money_series.iat[window] / 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data[window:], label='Asset Price', color='blue', alpha=0.7)
    plt.plot(sma_series[window:], label=f'SMA ({window}-day)', color='green', linestyle='--')
    plt.plot(money_series[window:], label='Cumulative Return', color='red')
    
    plt.title(f"{window}-Day SMA Crossover Strategy: {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (Starting at 100)')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

visualize_SMA_crossover('UUP', window=200, 
    buffer_start=0.98, buffer_end=1.02, min_hold=0, max_hold=1)
# visualize_EWMA_crossover('INTC', beta1=0, beta2=0.99)


