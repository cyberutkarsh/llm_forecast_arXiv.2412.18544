#!/usr/bin/env python3
"""
Time-bounded stock data retriever for backtesting.
"""

import yfinance as yf
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import datetime
from .stock_data import StockDataRetriever

class BacktestStockDataRetriever(StockDataRetriever):
    """Retrieves historical stock market data for a specific cutoff date."""
    
    def __init__(self, cutoff_date: str):
        """
        Initialize the backtesting stock data retriever.
        
        Args:
            cutoff_date: Date in YYYY-MM-DD format representing the knowledge cutoff
        """
        self.cutoff_date = cutoff_date
        self.cutoff_datetime = datetime.datetime.strptime(cutoff_date, "%Y-%m-%d")
        print(f"[Backtest Stocks] Initialized with cutoff date: {cutoff_date}")
        
    def get_historical_price_at_date(self, symbol: str, date: str) -> Optional[float]:
        """
        Get the stock price on a specific historical date.
        
        Args:
            symbol: The stock symbol (e.g., 'AAPL', 'MSFT', 'AMD')
            date: The date in YYYY-MM-DD format
            
        Returns:
            Stock price on that date or None if not found
        """
        try:
            # Make sure the date doesn't exceed the cutoff
            target_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            if target_date > self.cutoff_datetime:
                print(f"[Backtest Stocks] Warning: Requested date {date} is after cutoff date {self.cutoff_date}")
                return None
                
            # Get historical data up to the cutoff date
            ticker = yf.Ticker(symbol)
            
            # Get one week of data around the target date to handle weekends/holidays
            start_date = (target_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = target_date.strftime("%Y-%m-%d")
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"[Backtest Stocks] No data found for {symbol} around {date}")
                return None
                
            # Get the last available price on or before the target date
            filtered_data = data[data.index <= pd.Timestamp(target_date)]
            if filtered_data.empty:
                print(f"[Backtest Stocks] No data found for {symbol} on or before {date}")
                return None
                
            last_price = filtered_data['Close'].iloc[-1]
            print(f"[Backtest Stocks] Found price for {symbol} on {filtered_data.index[-1].strftime('%Y-%m-%d')}: {last_price}")
            return last_price
            
        except Exception as e:
            print(f"[Backtest Stocks] Error retrieving historical price for {symbol} on {date}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Override to get the stock price as of the cutoff date instead of the current price.
        
        Args:
            symbol: The stock symbol (e.g., 'AAPL', 'MSFT', 'AMD')
            
        Returns:
            Stock price as of the cutoff date or None if not found
        """
        print(f"[Backtest Stocks] Getting price for {symbol} as of {self.cutoff_date}")
        
        # Get the price on the cutoff date
        price = self.get_historical_price_at_date(symbol, self.cutoff_date)
        
        # If no data for exact cutoff date, try the previous trading day
        if price is None:
            # Try to get the previous trading day
            cutoff_dt = datetime.datetime.strptime(self.cutoff_date, "%Y-%m-%d")
            prev_day = (cutoff_dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            price = self.get_historical_price_at_date(symbol, prev_day)
            
            # Try up to 5 days back to find a valid trading day
            days_back = 2
            while price is None and days_back <= 5:
                prev_day = (cutoff_dt - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
                price = self.get_historical_price_at_date(symbol, prev_day)
                days_back += 1
        
        if price is not None:
            print(f"[Backtest Stocks] Found historical price for {symbol} as of {self.cutoff_date}: {price}")
        else:
            print(f"[Backtest Stocks] Could not find historical price for {symbol} as of {self.cutoff_date}")
            
        return price 