"""
Example tool implementations to demonstrate the extensibility of the tool system.
These tools are provided as templates that you can use to create your own tools.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import random
from .base import Tool

class DateTimeTool(Tool):
    """Tool that provides the current date and time."""
    
    @property
    def name(self) -> str:
        return "current_datetime"
        
    @property
    def description(self) -> str:
        return "Get the current date and time. Useful for timestamp references or when timing information is needed."
    
    def _run(self) -> Dict[str, str]:
        """Return the current date and time in multiple formats."""
        now = datetime.now()
        return {
            "iso_format": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timezone": datetime.now().astimezone().tzname()
        }

class StockPriceTool(Tool):
    """
    Example tool that simulates retrieving stock prices.
    
    Note: This is a mock implementation for demonstration purposes.
    In a real implementation, you would connect to a financial data API.
    """
    
    @property
    def name(self) -> str:
        return "stock_price"
        
    @property
    def description(self) -> str:
        return "Retrieve the current price of a stock by its ticker symbol."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL, MSFT, NVDA)"
            }
        }
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Mock implementation to simulate getting a stock price.
        
        In a real implementation, you would:
        1. Use a financial API like Alpha Vantage, Yahoo Finance, etc.
        2. Make API requests with proper authentication
        3. Parse and return the actual stock data
        """
        # This is just mock data for demonstration
        mock_prices = {
            "AAPL": 187.5 + random.uniform(-5, 5),
            "MSFT": 420.0 + random.uniform(-8, 8),
            "NVDA": 116.0 + random.uniform(-4, 4),
            "GOOGL": 176.0 + random.uniform(-5, 5),
            "AMZN": 183.0 + random.uniform(-6, 6),
            "META": 500.0 + random.uniform(-10, 10),
            "TSLA": 178.0 + random.uniform(-7, 7),
        }
        
        ticker = ticker.upper()
        if ticker in mock_prices:
            price = mock_prices[ticker]
        else:
            # Generate a random price for unknown tickers
            price = 100.0 + random.uniform(-20, 20)
            
        return {
            "ticker": ticker,
            "price": round(price, 2),
            "currency": "USD",
            "timestamp": datetime.now().isoformat(),
            "note": "This is simulated data for demonstration purposes only."
        }

# Example of how to register these tools:
"""
from tools import default_registry
from tools.examples import DateTimeTool, StockPriceTool

# Register the tools with the default registry
default_registry.register(DateTimeTool())
default_registry.register(StockPriceTool())
""" 