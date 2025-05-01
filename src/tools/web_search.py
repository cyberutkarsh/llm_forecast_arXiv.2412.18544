import os
import json
import requests
from typing import Any, Dict, List, Optional
import datetime
import hashlib
import pickle
import time
import re
from pathlib import Path
import openai
from .base import Tool
from bs4 import BeautifulSoup
import traceback

class WebSearchTool(Tool):
    """Tool for searching the web using Google's Custom Search API and extracting information with LLMs."""
    
    def __init__(self, api_key: Optional[str] = None, cx: Optional[str] = None, 
                 num_results: int = 5, include_snippets: bool = True,
                 cache_dir: Optional[str] = ".search_cache",
                 llm_model: str = "gpt-4o"):
        """
        Initialize the web search tool.
        
        Args:
            api_key: Google API key. If None, will look for GOOGLE_API_KEY env variable
            cx: Google Custom Search Engine ID. If None, will look for GOOGLE_CX env variable
            num_results: Number of results to return (max 10)
            include_snippets: Whether to include snippets in results
            cache_dir: Directory to store search result cache, if None caching is disabled
            llm_model: The OpenAI model to use for information extraction
        """
        # API credentials
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cx = cx or os.getenv("GOOGLE_CX")
        
        if not self.api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY env variable or pass it as parameter.")
        if not self.cx:
            raise ValueError("Google Custom Search Engine ID not found. Please set GOOGLE_CX env variable or pass it as parameter.")
        
        # Search parameters    
        self.num_results = min(num_results, 10)  # Google CSE has a max of 10 results per query
        self.include_snippets = include_snippets
        self.llm_model = llm_model
        
        # LLM access
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Google Search API endpoint
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_expiry = 24 * 60 * 60  # Cache valid for 24 hours
    
    @property
    def name(self) -> str:
        return "web_search"
        
    @property
    def description(self) -> str:
        return "Search the web for real-time information about any topic. Useful for gathering facts outside the model's training data."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query to use"
            }
        }
    
    def _run(self, query: str) -> Dict[str, Any]:
        """
        Run a web search with the given query and extract information.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary containing search results and extracted information
        """
        print(f"[Search] Processing query: '{query}'")
        
        # 1. Check cache first
        cache_key = self._get_cache_key(query)
        cached_results = self._get_from_cache(cache_key)
        if cached_results:
            return cached_results
        
        # 2. Fetch search results from Google API
        search_results = self._fetch_search_results(query)
        
        # 3. Process results
        processed_results = self._process_search_results(search_results, query)
        
        # 4. Extract information using LLM
        if processed_results and processed_results.get("results"):
            extracted_info = self._extract_information_with_llm(processed_results["results"], query)
            processed_results["extracted_information"] = extracted_info
        
        # 5. Cache results
        self._save_to_cache(cache_key, processed_results)
        
        return processed_results
    
    def _fetch_search_results(self, query: str) -> Dict[str, Any]:
        """
        Fetch raw search results from Google Custom Search API.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        # Set up query parameters
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": self.num_results
        }
        
        print(f"[Search] Fetching web search results...")
        
        try:
            # Make the request
            response = requests.get(self.endpoint, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"[Search] API request failed: {response.status_code}")
                return {
                    "query": query,
                    "results": [],
                    "timestamp": current_date,
                    "success": False,
                    "error": f"API request failed with status {response.status_code}"
                }
            
            # Parse response
            data = response.json()
            
            if "items" not in data:
                print(f"[Search] No results found")
                return {
                    "query": query,
                    "results": [],
                    "timestamp": current_date,
                    "success": True
                }
            
            # Success
            return {
                "query": query,
                "raw_data": data,
                "timestamp": current_date,
                "success": True
            }
            
        except Exception as e:
            print(f"[Search] Error fetching search results: {e}")
            return {
                "query": query,
                "results": [],
                "timestamp": current_date,
                "success": False,
                "error": str(e)
            }
    
    def _process_search_results(self, search_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process raw search results into a structured format.
        
        Args:
            search_data: Raw search data from API
            query: Original search query
            
        Returns:
            Processed search results
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Handle failed searches
        if not search_data.get("success", False):
            return search_data
        
        # Handle empty results
        if "raw_data" not in search_data or "items" not in search_data["raw_data"]:
            return {
                "query": query,
                "results": [],
                "timestamp": current_date,
                "success": True
            }
        
        # Process results
        results = []
        for item in search_data["raw_data"]["items"]:
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "date": current_date
            }
            
            if self.include_snippets and "snippet" in item:
                result["snippet"] = item["snippet"]
                
            results.append(result)
        
        return {
            "query": query,
            "results": results,
            "timestamp": current_date,
            "success": True
        }
    
    def _extract_information_with_llm(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Use LLM to extract relevant information from search results.
        
        Args:
            results: Processed search results
            query: Original search query
            
        Returns:
            Dictionary with extracted information
        """
        if not results:
            return {"values": []}
        
        # Prepare search results for the LLM
        result_texts = []
        for idx, result in enumerate(results[:5]):  # Limit to first 5 results
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            source = result.get("link", "")
            result_texts.append(f"Result {idx+1}:\nTitle: {title}\nSnippet: {snippet}\nSource: {source}\n")
        
        results_text = "\n".join(result_texts)
        
        # Create prompt for information extraction
        prompt = f"""
        You are an expert at extracting information from search results.
        
        User Query: "{query}"
        
        Here are the search results:
        
        {results_text}
        
        Analyze these search results to extract information relevant to the user's query.
        
        Focus on:
        1. Extracting any numeric values (prices, statistics, percentages, dates)
        2. Finding key facts relevant to the query
        3. Identifying current information rather than historical data
        4. Extracting information from the most reliable sources first
        
        For any numeric values found, please include:
        - The exact value
        - The unit (e.g., dollars, percentage, points)
        - What entity it relates to
        - Confidence level (high/medium/low)
        - Source

        Return your findings as a JSON object with the following structure:
        {{
            "values": [
                {{
                    "value": extracted numeric value (as a number),
                    "unit": unit of measurement,
                    "entity": what this value relates to,
                    "description": brief description of what this value represents,
                    "source": where this information was found,
                    "confidence": "high", "medium", or "low"
                }},
                // Add more values if found
            ],
            "summary": "A brief summary of the key information found in the search results",
            "key_facts": ["fact 1", "fact 2", ...],
            "content_type": "stock", "cryptocurrency", "weather", "sports", "news", or "general"
        }}
        
        If the query appears to be about stocks, financial instruments, or cryptocurrency, make special effort to extract the current price.
        If no relevant information can be found, return an empty "values" array but still provide a summary of what the search results contain.
        """
        
        try:
            print("[Search] Using LLM to extract information from search results")
            
            # Call LLM
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Process response
            try:
                content = response.choices[0].message.content
                extracted_info = json.loads(content)
                
                # Log the extracted values
                if "values" in extracted_info and extracted_info["values"]:
                    print(f"[Search] Extracted {len(extracted_info['values'])} values")
                    for value in extracted_info["values"]:
                        print(f"[Search] {value.get('entity', 'Unknown')}: {value.get('value')} {value.get('unit', '')}")
                
                return extracted_info
                
            except json.JSONDecodeError:
                print("[Search] Error parsing LLM output as JSON")
                return {"values": [], "summary": "Error extracting information", "key_facts": []}
                
        except Exception as e:
            print(f"[Search] Error using LLM for information extraction: {e}")
            return {"values": [], "summary": f"Error: {str(e)}", "key_facts": []}
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Try to get results from cache."""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
            
        # Check if cache has expired
        file_time = cache_file.stat().st_mtime
        if time.time() - file_time > self.cache_expiry:
            print(f"[Search] Cache expired for key {cache_key}")
            return None
            
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                
                # Validate timestamp
                if "timestamp" in cached_data:
                    try:
                        cached_date = datetime.datetime.strptime(cached_data["timestamp"], "%Y-%m-%d")
                        now = datetime.datetime.now()
                        
                        # If timestamp is in the future or too old (> 7 days), invalidate
                        if cached_date > now or (now - cached_date).days > 7:
                            print(f"[Search] Cache has invalid timestamp: {cached_data['timestamp']}")
                            return None
                    except (ValueError, TypeError):
                        # If timestamp parsing fails, continue using cache
                        pass
                        
                print(f"[Search] Using cached results from {cached_data.get('timestamp', 'unknown date')}")
                cached_data["cached"] = True
                return cached_data
        except Exception as e:
            print(f"[Search] Error reading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, results: Dict) -> None:
        """Save results to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Warning: Failed to cache search results: {e}")
            
    def fetch_webpage_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract content from a webpage.
        
        Args:
            url: URL of the webpage
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"[Search] Failed to fetch webpage: HTTP {response.status_code}")
                return None
                
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"[Search] Error fetching webpage content: {e}")
            return None 