#!/usr/bin/env python3
"""
Utility for retrieving stock market data using yfinance.
"""

import yfinance as yf
from typing import Dict, Any, Optional, Union, List
import pandas as pd

class StockDataRetriever:
    """Retrieves stock market data using yfinance."""
    
    @staticmethod
    def get_current_price(symbol: str) -> Optional[float]:
        """
        Get the current price for a given stock symbol.
        
        Args:
            symbol: The stock symbol (e.g., 'AAPL', 'MSFT', 'AMD')
            
        Returns:
            Current stock price or None if not found
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try using fast_info.last_price first
            try:
                return ticker.fast_info.last_price
            except (AttributeError, KeyError):
                pass
            
            # Fall back to info dictionary if fast_info isn't available
            try:
                return ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
            except (AttributeError, KeyError):
                pass
            
            # Fall back to history method as last resort
            data = ticker.history(period='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
            
            return None
        except Exception as e:
            print(f"Error retrieving stock price for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_historical_prices(symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Get historical prices for a given stock symbol.
        
        Args:
            symbol: The stock symbol (e.g., 'AAPL', 'MSFT', 'AMD')
            period: The time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame of historical prices or None if not found
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            print(f"Error retrieving historical prices for {symbol}: {e}")
            return None
    
    @staticmethod
    def search_stock_symbol(company_name: str) -> Optional[str]:
        """
        Attempt to find the stock symbol for a company name.
        This is a simple implementation and might not work for all companies.
        
        Args:
            company_name: The name of the company
            
        Returns:
            Best matching stock symbol or None if not found
        """
        # Common ticker mappings
        ticker_mapping = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'tesla': 'TSLA',
            'nvidia': 'NVDA',
            'amd': 'AMD',
            'advanced micro devices': 'AMD',
            'intel': 'INTC',
            'meta': 'META',
            'facebook': 'META',
            'netflix': 'NFLX',
        }
        
        # Check if company name is in our mapping
        company_name_lower = company_name.lower()
        if company_name_lower in ticker_mapping:
            return ticker_mapping[company_name_lower]
        
        # Try to search using yfinance search functionality
        try:
            search_result = yf.Ticker(company_name)
            # If we can get info, it might be a valid ticker
            info = search_result.info
            if 'symbol' in info:
                return info['symbol']
        except:
            pass
        
        return None

if __name__ == "__main__":
    # Test the retriever
    retriever = StockDataRetriever()
    
    # Test current price retrieval
    symbols = ['AAPL', 'MSFT', 'AMD', 'NVDA']
    for symbol in symbols:
        price = retriever.get_current_price(symbol)
        print(f"{symbol} current price: {price}")
    
    # Test symbol search
    companies = ['Apple', 'Microsoft', 'AMD', 'Advanced Micro Devices']
    for company in companies:
        symbol = retriever.search_stock_symbol(company)
        print(f"{company} => {symbol}") 