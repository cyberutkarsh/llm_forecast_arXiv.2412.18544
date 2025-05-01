import os
import json
import requests
from typing import Any, Dict, Optional, List
import datetime
import time
from pathlib import Path
from .base import Tool

class CryptoDataTool(Tool):
    """Tool for retrieving current cryptocurrency data using CoinGecko API."""
    
    def __init__(self, cache_dir: Optional[str] = ".crypto_cache"):
        """
        Initialize the crypto data tool.
        
        Args:
            cache_dir: Directory to store cache, if None caching is disabled
        """
        # API setup - CoinGecko's free tier doesn't require API key
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_expiry = 5 * 60  # Cache valid for 5 minutes (crypto prices change rapidly)
        
        # Load coin list from CoinGecko or cache
        self.coin_list = self._get_coin_list()
    
    @property
    def name(self) -> str:
        return "crypto_data"
        
    @property
    def description(self) -> str:
        return "Retrieves current cryptocurrency data using CoinGecko API"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "crypto_id": {
                "type": "string",
                "description": "The cryptocurrency ID, symbol, or name (e.g., bitcoin, BTC, Ethereum)"
            }
        }
    
    def _run(self, crypto_id: str) -> Dict[str, Any]:
        """
        Get current crypto data for the specified cryptocurrency.
        
        Args:
            crypto_id: The cryptocurrency ID, symbol, or name
            
        Returns:
            Dictionary with cryptocurrency data
        """
        # Standardize format
        crypto_id = crypto_id.strip().lower()
        print(f"[Crypto] Getting data for crypto: {crypto_id}")
        
        # Try to get coingecko_id from input
        coingecko_id = self._get_coingecko_id(crypto_id)
        if not coingecko_id:
            return {
                "success": False,
                "error": f"Could not find cryptocurrency with ID, symbol, or name: {crypto_id}",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Check cache first
        cache_key = coingecko_id
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            print(f"[Crypto] Using cached data for {coingecko_id}")
            return cached_data
        
        # Fetch from CoinGecko API
        crypto_data = self._fetch_crypto_data(coingecko_id)
        
        if crypto_data.get("success", False):
            # Cache the data
            self._save_to_cache(cache_key, crypto_data)
            return crypto_data
        
        # API request failed
        return {
            "success": False,
            "crypto_id": crypto_id,
            "coingecko_id": coingecko_id,
            "error": "Could not retrieve crypto data from CoinGecko API",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _get_coin_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all coins from CoinGecko API or cache.
        
        Returns:
            List of coin dictionaries
        """
        # Check if we have a cached version
        if self.cache_dir:
            cache_file = self.cache_dir / "coin_list.json"
            
            if cache_file.exists():
                file_time = cache_file.stat().st_mtime
                # Coin list cache valid for 24 hours
                if time.time() - file_time < 24 * 60 * 60:
                    try:
                        with open(cache_file, "r") as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"[Crypto] Error reading coin list cache: {e}")
        
        # Fetch from API
        try:
            print("[Crypto] Fetching coin list from CoinGecko API")
            response = requests.get(f"{self.base_url}/coins/list", timeout=10)
            
            if response.status_code != 200:
                print(f"[Crypto] Failed to fetch coin list: HTTP {response.status_code}")
                return []
            
            coin_list = response.json()
            
            # Cache the coin list
            if self.cache_dir:
                try:
                    with open(self.cache_dir / "coin_list.json", "w") as f:
                        json.dump(coin_list, f)
                except Exception as e:
                    print(f"[Crypto] Error saving coin list cache: {e}")
            
            return coin_list
            
        except Exception as e:
            print(f"[Crypto] Error fetching coin list: {e}")
            return []
    
    def _get_coingecko_id(self, crypto_input: str) -> Optional[str]:
        """
        Get the CoinGecko ID for a given cryptocurrency input.
        
        Args:
            crypto_input: The cryptocurrency ID, symbol, or name
            
        Returns:
            CoinGecko ID or None if not found
        """
        # If the input is already a valid ID, return it
        for coin in self.coin_list:
            if coin.get("id") == crypto_input:
                return crypto_input
        
        # Try to match by symbol (e.g., BTC, ETH)
        symbol_matches = [coin for coin in self.coin_list if coin.get("symbol", "").lower() == crypto_input.lower()]
        if symbol_matches:
            # Prioritize popular coins if there are multiple matches
            popular_coins = ["bitcoin", "ethereum", "ripple", "cardano", "solana", "dogecoin"]
            for popular in popular_coins:
                for coin in symbol_matches:
                    if coin.get("id") == popular:
                        return popular
            # Return the first match if no popular coin is found
            return symbol_matches[0].get("id")
        
        # Try to match by name (e.g., Bitcoin, Ethereum)
        name_matches = [coin for coin in self.coin_list if coin.get("name", "").lower() == crypto_input.lower()]
        if name_matches:
            return name_matches[0].get("id")
        
        # Try partial name match as a last resort
        partial_matches = [coin for coin in self.coin_list if crypto_input.lower() in coin.get("name", "").lower()]
        if partial_matches:
            return partial_matches[0].get("id")
        
        return None
    
    def _fetch_crypto_data(self, coingecko_id: str) -> Dict[str, Any]:
        """
        Fetch cryptocurrency data from CoinGecko API.
        
        Args:
            coingecko_id: CoinGecko coin ID
            
        Returns:
            Dictionary with crypto data
        """
        try:
            print(f"[Crypto] Fetching {coingecko_id} data from CoinGecko API")
            
            # Set parameters for API call
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false"
            }
            
            # Make the API request
            response = requests.get(
                f"{self.base_url}/coins/{coingecko_id}",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"[Crypto] Failed to fetch CoinGecko data: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            
            # Extract relevant data
            market_data = data.get("market_data", {})
            current_price_usd = market_data.get("current_price", {}).get("usd")
            
            if not current_price_usd:
                print(f"[Crypto] No price data found for {coingecko_id}")
                return {"success": False, "error": "No price data found"}
            
            # Create result object
            result = {
                "success": True,
                "coingecko_id": coingecko_id,
                "name": data.get("name", coingecko_id),
                "symbol": data.get("symbol", "").upper(),
                "price": current_price_usd,
                "price_unit": "USD",
                "market_cap": market_data.get("market_cap", {}).get("usd"),
                "percent_change_24h": market_data.get("price_change_percentage_24h"),
                "volume_24h": market_data.get("total_volume", {}).get("usd"),
                "high_24h": market_data.get("high_24h", {}).get("usd"),
                "low_24h": market_data.get("low_24h", {}).get("usd"),
                "rank": data.get("market_cap_rank"),
                "source": "CoinGecko API",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"[Crypto] Successfully extracted {coingecko_id} data: ${current_price_usd}")
            return result
            
        except Exception as e:
            print(f"[Crypto] Error fetching from CoinGecko API: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get crypto data from cache if available and not expired."""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        # Check if cache has expired
        file_time = cache_file.stat().st_mtime
        if time.time() - file_time > self.cache_expiry:
            print(f"[Crypto] Cache expired for {cache_key}")
            return None
            
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                cached_data["cached"] = True
                return cached_data
        except Exception as e:
            print(f"[Crypto] Error reading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save crypto data to cache."""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Crypto] Error saving to cache: {e}") 