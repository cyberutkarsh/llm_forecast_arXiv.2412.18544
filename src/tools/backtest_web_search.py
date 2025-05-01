import os
import datetime
from typing import Dict, Any, Optional
from .web_search import WebSearchTool
import re

class BacktestWebSearchTool(WebSearchTool):
    """Time-bounded web search tool for backtesting forecasts."""
    
    def __init__(self, cutoff_date: str, api_key: Optional[str] = None, cx: Optional[str] = None,
                 num_results: int = 5, include_snippets: bool = True, cache_dir: str = ".search_cache"):
        """
        Initialize the backtest web search tool.
        
        Args:
            cutoff_date: Date in YYYY-MM-DD format representing the knowledge cutoff
            api_key: Google API key. If None, will look for GOOGLE_API_KEY env variable
            cx: Google Custom Search Engine ID. If None, will look for GOOGLE_CX env variable
            num_results: Number of results to return (max 10)
            include_snippets: Whether to include snippets in results
            cache_dir: Directory to store search result cache
        """
        super().__init__(api_key, cx, num_results, include_snippets, cache_dir)
        self.cutoff_date = cutoff_date
        # Append a timestamp to the name to make it distinct
        self._name = f"backtest_web_search_{cutoff_date}"
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def description(self) -> str:
        return f"Search the web for information available before {self.cutoff_date}. " + \
               "Results are time-filtered to exclude information after this date."
    
    def _run(self, query: str) -> Dict[str, Any]:
        """
        Run a time-bounded web search with the given query.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary containing search results
        """
        # Modify query to include date restriction
        date_restricted_query = f"{query} before:{self.cutoff_date}"
        print(f"[Backtest Search] Using time-bounded query: '{date_restricted_query}'")
        
        # Run the search with the modified query
        results = super()._run(date_restricted_query)
        
        # Add metadata about the backtest
        results["backtest_mode"] = True
        results["cutoff_date"] = self.cutoff_date
        
        # Filter out any results that might mention dates after the cutoff
        # This is a simple extra precaution
        if "results" in results:
            filtered_results = []
            cutoff_dt = datetime.datetime.strptime(self.cutoff_date, "%Y-%m-%d")
            
            for result in results["results"]:
                # Skip results that explicitly mention later dates in the title or snippet
                if self._contains_future_date(result.get("title", ""), cutoff_dt) or \
                   self._contains_future_date(result.get("snippet", ""), cutoff_dt):
                    continue
                filtered_results.append(result)
            
            results["results"] = filtered_results
            results["filtered_count"] = len(results["results"]) - len(filtered_results)
        
        return results
    
    def _contains_future_date(self, text: str, cutoff_dt: datetime.datetime) -> bool:
        """
        Check if the text contains dates after the cutoff date.
        This is a simple check and may not catch all date formats.
        
        Args:
            text: The text to check
            cutoff_dt: The cutoff date as a datetime object
            
        Returns:
            True if future dates are found, False otherwise
        """
        # Check for common date formats: YYYY-MM-DD, MM/DD/YYYY, etc.
        # This is a simple implementation and can be expanded
        date_patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                try:
                    if pattern.startswith(r'(\d{4})'):
                        year, month, day = match.groups()
                    else:
                        month, day, year = match.groups()
                    
                    date = datetime.datetime(int(year), int(month), int(day))
                    if date > cutoff_dt:
                        return True
                except (ValueError, IndexError):
                    # Invalid date format, skip
                    pass
        
        return False 