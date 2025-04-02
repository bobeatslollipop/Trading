import yfinance as yf
import pandas as pd
import numpy as np
import gurobipy as GRB
import matplotlib.pyplot as plt
from asset_analysis import calculate_daily_changes, fetch_ticker
import math


def ewma_crossover(data: pd.DataFrame, beta1=0.95, beta2=0.99):
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


def visualize_ewma_crossover(ticker:str, beta1=0.95, beta2=0.99, start_date ="2005-01-01", end_date ="2025-03-27"):
    data = fetch_ticker(ticker)
    money, ma1, ma2 = ewma_crossover(data, beta1, beta2)

    # Convert to series for easy plotting
    ma1_series = pd.Series(ma1, index=data.index)
    ma2_series = pd.Series(ma2, index=data.index)
    money_series = pd.Series(money, index=data.index)

    # Normalize data to start at 100 for better comparison
    data = data / data.iloc[0] * 100
    ma1_series = ma1_series / ma1_series.iloc[0] * 100
    ma2_series = ma2_series / ma2_series.iloc[0] * 100

    # Plot all lines on the same axis
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Asset Price', color='blue')
    plt.plot(ma1_series, label=f'ma1 (β={beta1})', color='orange', linestyle='--')
    plt.plot(ma2_series, label=f'ma2 (β={beta2})', color='green', linestyle='--')
    plt.plot(money_series, label='Cumulative Return', color='red')

    plt.title(f"EWMAC Strategy for {ticker}")
    plt.xlabel('Date')
    plt.yscale('log')
    plt.ylabel('Normalized Value (Starting at 100)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()


def sma_crossover(data: pd.DataFrame, window=200,
                  buffer_start=1, buffer_end=1, min_hold=0, max_hold=1):
    """
    Implements a Simple Moving Average (SMA) crossover trading strategy.

    Args:
        data (pd.DataFrame): Price data (single-column DataFrame with asset prices).
        window (int): Lookback window for SMA computation (default: 200-day).
        buffer_start (float): Lower threshold for transitioning into a position.
        buffer_end (float): Upper threshold for transitioning out of a position.
        min_hold (float): Minimum holding position (0 = no position).
        max_hold (float): Maximum holding position (1 = fully invested).

    Returns:
        np.array: Cumulative returns over time, starting at 100.
        np.array: Computed SMA values.
        np.array: Holding positions over time.
    """
    sma = data.rolling(window=window).mean().values.flatten()
    positions = [0]
    equity = [100]  # Start with an initial equity value of 100

    for i in range(1, len(data)):
        # Update equity based on previous position and today's price change
        current_equity = equity[-1] * ((1 - positions[-1]) * data.values[i - 1, 0] +
                                       positions[-1] * data.values[i, 0]) / data.values[i - 1, 0]
        equity.append(current_equity)

        ratio = data.values[i, 0] / sma[i]

        if math.isnan(sma[i]):  # SMA window not yet valid
            positions.append(0)
        elif buffer_start == buffer_end:  # No buffer zone (standard SMA crossover)
            positions.append(int(data.values[i, 0] >= sma[i]) * (max_hold - min_hold) + min_hold)
        elif ratio >= buffer_end:
            positions.append(max_hold)
        elif ratio <= buffer_start:
            positions.append(min_hold)
        elif ratio <= 1:
            positions.append((ratio - buffer_start) / (1 - buffer_start) * min_hold)
        else:  # ratio between 1 and buffer_end
            positions.append((buffer_end - ratio) / (buffer_end - 1) * max_hold)

    return np.array(equity), sma, np.array(positions)


def visualize_sma_crossover(ticker: str, window=200,
                            start_date="2005-01-01", end_date="2025-03-27",
                            buffer_start=1, buffer_end=1, min_hold=0, max_hold=1):
    """
    Visualizes the performance of an SMA crossover strategy against the underlying asset.

    Args:
        ticker (str): Asset ticker symbol.
        window (int): SMA window period.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        buffer_start (float): Entry buffer below SMA.
        buffer_end (float): Exit buffer above SMA.
        min_hold (float): Minimum position.
        max_hold (float): Maximum position.
    """
    data = fetch_ticker(ticker)
    data = data.loc[start_date:end_date]

    equity, sma, _ = sma_crossover(data, window,
                                   buffer_start=buffer_start,
                                   buffer_end=buffer_end,
                                   min_hold=min_hold,
                                   max_hold=max_hold)

    sma_series = pd.Series(sma, index=data.index)
    equity_series = pd.Series(equity, index=data.index)

    # Normalize data for plotting
    normalization_factor = data.iat[window, 0] / 100
    sma_series /= normalization_factor
    data /= normalization_factor
    equity_series /= equity_series.iat[window] / 100

    # Plot strategy results
    plt.figure(figsize=(12, 6))
    plt.plot(data[window:], label='Asset Price', color='blue', alpha=0.7)
    plt.plot(sma_series[window:], label=f'{window}-day SMA', color='green', linestyle='--')
    plt.plot(equity_series[window:], label='Strategy Cumulative Return', color='red')

    plt.title(f"{window}-Day SMA Crossover Strategy for {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (Base = 100)')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

# TODO: 网格法，（均值回归？）

visualize_sma_crossover('^GSPC', window=200,
                        buffer_start=0.98, buffer_end=1.02, min_hold=0, max_hold=1)
# visualize_EWMA_crossover('INTC', beta1=0, beta2=0.99)


ticker_list = ['SPY', 'QQQ', 'GLD', 'VGLT', 'UUP', 'FXI']