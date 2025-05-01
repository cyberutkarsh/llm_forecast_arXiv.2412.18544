import os
import json
import requests
from typing import Any, Dict, Optional
import datetime
import time
from pathlib import Path
from .base import Tool

class StockDataTool(Tool):
    """Tool for retrieving current stock data using Alpha Vantage API."""
    
    def __init__(self, cache_dir: Optional[str] = ".stock_cache", api_key: Optional[str] = None):
        """
        Initialize the stock data tool.
        
        Args:
            cache_dir: Directory to store cache, if None caching is disabled
            api_key: Alpha Vantage API key, if None will look for ALPHA_VANTAGE_API_KEY env variable
        """
        # API setup
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY env variable or pass it as parameter.")
        
        self.base_url = "https://www.alphavantage.co/query"
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_expiry = 15 * 60  # Cache valid for 15 minutes (stock data changes quickly)
    
    @property
    def name(self) -> str:
        return "stock_data"
        
    @property
    def description(self) -> str:
        return "Retrieves current stock data for a specific ticker symbol using Alpha Vantage API"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
            }
        }
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Get current stock data for the specified ticker.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dictionary with stock data
        """
        # Standardize ticker format
        ticker = ticker.strip().upper()
        print(f"[Stock] Getting data for ticker: {ticker}")
        
        # Check cache first
        cache_key = f"{ticker.lower()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            print(f"[Stock] Using cached data for {ticker}")
            return cached_data
        
        # Fetch from Alpha Vantage API
        quote_data = self._fetch_global_quote(ticker)
        
        # If quote data available, use that for current price and change
        if quote_data.get("success", False):
            # Cache the data
            self._save_to_cache(cache_key, quote_data)
            return quote_data
        
        # If quote data fails, try overview for company name, etc.
        overview_data = self._fetch_company_overview(ticker)
        if overview_data.get("success", False):
            # Combine data if possible
            if quote_data.get("success", False):
                combined_data = {**overview_data, **quote_data}
                combined_data["success"] = True
                # Cache the combined data
                self._save_to_cache(cache_key, combined_data)
                return combined_data
            else:
                # Cache the overview data
                self._save_to_cache(cache_key, overview_data)
                return overview_data
        
        # Both failed, return error
        return {
            "success": False,
            "ticker": ticker,
            "error": "Could not retrieve stock data from Alpha Vantage API",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _fetch_global_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch global quote data from Alpha Vantage API.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock data
        """
        try:
            print(f"[Stock] Fetching {ticker} data from Alpha Vantage (Global Quote)")
            
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"[Stock] Failed to fetch Alpha Vantage data: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            
            # Check if we got an error response
            if "Error Message" in data:
                print(f"[Stock] Alpha Vantage API error: {data['Error Message']}")
                return {"success": False, "error": data["Error Message"]}
                
            # Check if we got a valid quote
            if "Global Quote" not in data or not data["Global Quote"]:
                print(f"[Stock] No quote data found for {ticker}")
                return {"success": False, "error": "No quote data found"}
            
            quote = data["Global Quote"]
            
            # Extract the data
            current_price = float(quote.get("05. price", 0))
            percent_change = float(quote.get("10. change percent", "0%").replace("%", ""))
            
            # Create result object
            result = {
                "ticker": ticker,
                "price": current_price,
                "percent_change": percent_change,
                "day_range": f"{float(quote.get('04. low', 0))}-{float(quote.get('03. high', 0))}",
                "source": "Alpha Vantage API",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
            print(f"[Stock] Successfully extracted {ticker} data: ${current_price}")
            return result
            
        except Exception as e:
            print(f"[Stock] Error fetching from Alpha Vantage (Global Quote): {e}")
            return {"success": False, "error": str(e)}
    
    def _fetch_company_overview(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company overview data from Alpha Vantage API.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company data
        """
        try:
            print(f"[Stock] Fetching {ticker} company data from Alpha Vantage")
            
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"[Stock] Failed to fetch Alpha Vantage company data: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            
            # Check if we got an error response
            if "Error Message" in data:
                print(f"[Stock] Alpha Vantage API error: {data['Error Message']}")
                return {"success": False, "error": data["Error Message"]}
            
            # Check if we got valid data (look for symbol or name)
            if not data.get("Symbol") and not data.get("Name"):
                print(f"[Stock] No company data found for {ticker}")
                return {"success": False, "error": "No company data found"}
            
            # Extract the data
            company_name = data.get("Name", ticker)
            sector = data.get("Sector", "")
            industry = data.get("Industry", "")
            market_cap = data.get("MarketCapitalization", "")
            
            # Create result object
            result = {
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
                "market_cap": market_cap,
                "source": "Alpha Vantage API",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
            print(f"[Stock] Successfully extracted {ticker} company data")
            return result
            
        except Exception as e:
            print(f"[Stock] Error fetching from Alpha Vantage (Company Overview): {e}")
            return {"success": False, "error": str(e)}
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get stock data from cache if available and not expired."""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        # Check if cache has expired (stock data expires quickly)
        file_time = cache_file.stat().st_mtime
        if time.time() - file_time > self.cache_expiry:
            print(f"[Stock] Cache expired for {cache_key}")
            return None
            
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                cached_data["cached"] = True
                return cached_data
        except Exception as e:
            print(f"[Stock] Error reading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save stock data to cache."""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Stock] Error saving to cache: {e}") 