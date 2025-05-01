import os
from typing import Dict, List, Any, Optional
import openai
import json
import random
import re
import concurrent.futures
import hashlib
import pickle
import time
from pathlib import Path
from tools import ToolRegistry, WebSearchTool, default_registry
from tools.backtest_registry import BacktestToolRegistry
from utils.question_analyzer import QuestionAnalyzer
from utils.stock_data import StockDataRetriever
from utils.backtest_stock_data import BacktestStockDataRetriever

class ForecastingSystem:
    def __init__(self, model="o3", tools_registry=None, cache_dir=".forecast_cache", max_retries=3, max_workers=4, analyzer_model="gpt-4o", disable_cache=False):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Set up question analyzer
        self.question_analyzer = QuestionAnalyzer(model=analyzer_model)
        
        # Handle cache_dir being None or disabled
        if disable_cache:
            cache_dir = None
        
        # Set up tools registry
        self.tools_registry = tools_registry or default_registry
        
        # Set up API retries
        self.max_retries = max_retries
        
        # Set up parallelization
        self.max_workers = max_workers
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.response_cache = {}
            self.cache_expiry = 24 * 60 * 60  # Cache valid for 24 hours
        
        # Set up shared context for consistency between checks
        self.shared_context = {}
        
        # Set up optimization stats tracking
        self.reset_stats()
        
        # Try to set up web search if environment variables are available
        try:
            # Set cache_dir to None if cache is disabled
            tools_cache_dir = None if cache_dir is None else cache_dir
            
            # 1. Web search tool
            if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CX"):
                web_search = WebSearchTool(cache_dir=f"{tools_cache_dir}/.search_cache", llm_model="gpt-4o")
                self.tools_registry.register(web_search)
                self._has_web_search = True
                print("[Setup] Web search tool initialized")
            else:
                self._has_web_search = False
                print("[Setup] Web search tool unavailable: Missing GOOGLE_API_KEY or GOOGLE_CX")
            
            # 2. Stock data tool using Alpha Vantage
            if os.getenv("ALPHA_VANTAGE_API_KEY"):
                from tools.stock_data import StockDataTool
                stock_data_tool = StockDataTool(cache_dir=f"{tools_cache_dir}/.stock_cache")
                self.tools_registry.register(stock_data_tool)
                self._has_stock_data = True
                print("[Setup] Stock data tool initialized")
            else:
                self._has_stock_data = False
                print("[Setup] Stock data tool unavailable: Missing ALPHA_VANTAGE_API_KEY")
            
            # 3. Crypto data tool using CoinGecko
            try:
                from tools.crypto_data import CryptoDataTool
                crypto_data_tool = CryptoDataTool(cache_dir=f"{tools_cache_dir}/.crypto_cache")
                self.tools_registry.register(crypto_data_tool)
                self._has_crypto_data = True
                print("[Setup] Crypto data tool initialized")
            except Exception as e:
                self._has_crypto_data = False
                print(f"[Setup] Crypto data tool initialization failed: {e}")
            
            # 4. Query classifier
            try:
                from tools.query_classifier import QueryClassifier
                self.query_classifier = QueryClassifier(model="o3")
                print("[Setup] Query classifier initialized")
            except Exception as e:
                print(f"[Setup] Query classifier initialization failed: {e}")
            
        except Exception as e:
            self._has_web_search = False
            self._has_stock_data = False
            self._has_crypto_data = False
            print(f"[Setup] Tool initialization error: {e}")
        
    def reset_stats(self):
        """Reset optimization statistics."""
        self.stats = {
            "cache_hits": 0,
            "api_calls": 0,
            "start_time": time.time(),
            "sequential_time": 0
        }
    
    def generate_forecast(self, question: str, num_checks: int = 3, verbose: bool = False, progress_callback=None) -> Dict[str, Any]:
        """
        Generate a forecast with consistency checks
        
        Args:
            question: The forecasting question
            num_checks: Number of consistency checks to perform
            verbose: Whether to include detailed reasoning
            progress_callback: Optional callback function to track progress
            
        Returns:
            Dictionary containing the forecast and consistency data
        """
        # Reset statistics
        self.reset_stats()
        
        # Analyze question to understand what's being asked
        print(f"[Analysis] Analyzing question: '{question}'")
        question_analysis = self.question_analyzer.analyze_question(question)
        
        if question_analysis.get("analysis_success"):
            print(f"[Analysis] Question type: {question_analysis.get('question_type', 'general')}")
            for key, value in question_analysis.items():
                if key not in ["analysis_success", "question_type"] and value:
                    print(f"  - {key}: {value}")
            
            # Store in shared context for consistency
            self.shared_context['question_analysis'] = question_analysis
            
            # Generate optimized search query if needed
            if self._has_web_search:
                optimized_query = self.question_analyzer.get_search_query(question_analysis)
                if optimized_query:
                    self.shared_context['optimized_query'] = optimized_query
                    print(f"[Analysis] Optimized search query: '{optimized_query}'")
        else:
            print(f"[Analysis] Could not analyze question: {question_analysis.get('error', 'Unknown error')}")
        
        # Time tracking for parallelization comparison
        sequential_start = time.time()
        
        # Initial forecast
        initial_result = self._get_direct_forecast(question)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback()
        
        # Generate check types first
        check_types = []
        for _ in range(num_checks):
            check_type = random.choice([
                "temporal_decomposition",
                "spatial_decomposition", 
                "chain_of_thought",
                "alternative_framing"
            ])
            check_types.append(check_type)
        
        # Run consistency checks in parallel
        consistency_checks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_checks, self.max_workers)) as executor:
            # Submit all tasks
            future_to_check_type = {
                executor.submit(self._run_consistency_check, question, check_type): check_type
                for check_type in check_types
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_check_type):
                check_type = future_to_check_type[future]
                try:
                    check_result = future.result()
                    consistency_checks.append(check_result)
                    
                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback()
                except Exception as e:
                    print(f"Error running {check_type} check: {e}")
                    # Create a fallback check result
                    consistency_checks.append({
                        "type": check_type,
                        "probability": 0.5,
                        "reasoning": f"Error: {str(e)}"
                    })
                    
                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback()
        
        # Calculate how long it would have taken sequentially
        sequential_time = time.time() - sequential_start
        
        # Calculate what it would have taken to run sequentially (approximation)
        sequential_estimate = sequential_time * (num_checks + 1) / (num_checks + 1 - len(consistency_checks))
        parallel_time_saved = sequential_estimate - sequential_time
        
        # Detect inconsistent interpretations
        if (question_analysis.get("analysis_success") and
            question_analysis.get("threshold") is not None and
            "extracted_values" in self.shared_context):
            
            # Check if there are widely divergent probabilities
            probabilities = [self._extract_probability(initial_result["probability"])] + [self._extract_probability(check["probability"]) for check in consistency_checks]
            
            if max(probabilities) - min(probabilities) > 0.6:
                # Get relevant values from context
                threshold = question_analysis.get("threshold")
                extracted_values = self.shared_context.get("extracted_values", {}).get("values", [])
                
                # Find most relevant value (matching unit if possible)
                relevant_value = None
                if "threshold_unit" in question_analysis and question_analysis["threshold_unit"]:
                    unit = question_analysis["threshold_unit"]
                    for value_info in extracted_values:
                        if value_info.get("unit") == unit:
                            relevant_value = value_info
                            break
                
                # If we found a value, add clarification
                if relevant_value:
                    value = relevant_value.get("value")
                    unit = relevant_value.get("unit")
                    comparison = "above" if value > threshold else "below"
                    
                    clarification = f"\n\nClarification: Based on the latest data, the current value is {value} {unit}, which is {comparison} the threshold of {threshold} {question_analysis.get('threshold_unit', '')}."
                    
                    if "forecast" in initial_result:
                        initial_result["forecast"] += clarification
                        
                    print(f"[Analysis] Added value clarification due to potential interpretation inconsistency")
        
        # Calculate overall consistency
        probabilities = [self._extract_probability(initial_result["probability"])] + [self._extract_probability(check["probability"]) for check in consistency_checks]
        consistency_score = self._calculate_consistency_score(probabilities)
        
        # Adjust forecast based on consistency
        final_forecast = initial_result["forecast"]
        confidence = "High" if consistency_score > 0.85 else "Medium" if consistency_score > 0.7 else "Low"
        
        # Collect optimization statistics
        self.stats["parallel_time_saved"] = parallel_time_saved
        self.stats["total_time"] = time.time() - self.stats["start_time"]
        
        return {
            "forecast": final_forecast,
            "initial_probability": initial_result["probability"],
            "confidence": confidence,
            "consistency_checks": consistency_checks,
            "consistency_score": consistency_score,
            "research": initial_result.get("research", []),
            "key_factors": initial_result.get("key_factors", ""),
            "confidence_explanation": initial_result.get("confidence_explanation", ""),
            "optimization_stats": self.stats,
            "question_analysis": question_analysis if question_analysis.get("analysis_success") else None
        }
    
    def _extract_probability(self, prob_string: str) -> float:
        """
        Extract a numeric probability value from a string using LLM.
        
        Args:
            prob_string: String that may contain a probability value
            
        Returns:
            Extracted probability as a float between 0 and 1
        """
        # If None is passed, return default
        if prob_string is None:
            print("Warning: Received None probability value. Using default of 0.5.")
            return 0.5
            
        # If it's already a number, just return it
        if isinstance(prob_string, (int, float)):
            if 0 <= prob_string <= 1:
                return float(prob_string)
            elif 0 <= prob_string <= 100:
                return float(prob_string) / 100
            else:
                print(f"Warning: Probability value {prob_string} out of range. Using default of 0.5.")
                return 0.5
        
        # Use LLM to extract the probability
        prompt = f"""
        Extract the numerical probability from this text. Return ONLY a decimal number between 0 and 1.
        
        Text: "{prob_string}"
        
        Rules for extraction:
        1. Return a single decimal number between 0 and 1 (e.g., 0.75)
        2. If a percentage is given (e.g., 75%), convert it to a decimal (0.75)
        3. If the probability refers to the event NOT happening, use the complement (1 - probability)
        4. If no clear probability is found, return 0.5
        5. If multiple probabilities are given, use the final or most specific one
        
        Examples:
        - "I estimate there's a 70% chance" → 0.7
        - "The probability is 0.25" → 0.25
        - "There's a 20% chance this will not happen" → 0.8
        - "Unlikely, with only about 30% probability" → 0.3
        
        Output ONLY the decimal number without any explanation or additional text.
        """
        
        try:
            # Use a more efficient and cost-effective model for simple extraction
            extraction_model = "gpt-4o"
            
            response = self.client.chat.completions.create(
                model=extraction_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to convert the result to float
            try:
                value = float(result)
                # Validate the value is in the proper range
                if 0 <= value <= 1:
                    return value
                elif 0 <= value <= 100:  # Sometimes it might still return a percentage
                    return value / 100
                else:
                    print(f"Warning: LLM returned out-of-range value '{value}'. Using default of 0.5.")
                    return 0.5
            except ValueError:
                print(f"Warning: Could not convert LLM result '{result}' to float. Using default of 0.5.")
                return 0.5
                
        except Exception as e:
            print(f"Warning: LLM probability extraction failed: {e}. Using default of 0.5.")
            return 0.5
        
    def _call_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with the given inputs."""
        tool = self.tools_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        return tool.run(**tool_input)
    
    def _call_api_with_retries(self, **kwargs):
        """Call OpenAI API with retries for transient errors."""
        self.stats["api_calls"] += 1
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
                last_exception = e
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f"API request failed, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        # If all retries failed, raise the last exception
        raise last_exception
    
    def _get_cache_key(self, messages, tools=None):
        """Generate a cache key for API request."""
        # If caching is disabled, return None
        if not self.cache_dir:
            return None
            
        # Serialize messages and tools to JSON
        key_data = {"messages": messages}
        if tools:
            key_data["tools"] = tools
        
        # Create a deterministic JSON string
        json_str = json.dumps(key_data, sort_keys=True)
        
        # Hash the string
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key):
        """Try to get response from cache."""
        # If caching is disabled or no cache key, return None
        if not self.cache_dir or not cache_key:
            return None
            
        # First check in-memory cache
        if cache_key in self.response_cache:
            self.stats["cache_hits"] += 1
            return self.response_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check if cache has expired
        file_time = cache_file.stat().st_mtime
        if time.time() - file_time > self.cache_expiry:
            return None
        
        try:
            with open(cache_file, "rb") as f:
                response = pickle.load(f)
                # Add to in-memory cache for faster access next time
                self.response_cache[cache_key] = response
                self.stats["cache_hits"] += 1
                return response
        except Exception:
            return None
    
    def _save_to_cache(self, cache_key, response):
        """Save response to cache."""
        # If caching is disabled or no cache key, do nothing
        if not self.cache_dir or not cache_key:
            return
            
        # Save to in-memory cache
        self.response_cache[cache_key] = response
        
        # Save to disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(response, f)
        except Exception as e:
            print(f"Warning: Failed to cache API response: {e}")
    
    def _get_direct_forecast(self, question: str) -> Dict[str, Any]:
        """Get a direct forecast from the LLM with tool calling if needed."""
        # Create messages for the API call
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self._get_user_prompt(question)}
        ]
        
        # Try to get cached response first (only for non-tool requests)
        tools = self.tools_registry.get_tool_descriptions() if self._has_web_search else None
        if not tools:
            cache_key = self._get_cache_key(messages)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                try:
                    return json.loads(cached_response.choices[0].message.content)
                except json.JSONDecodeError as e:
                    print(f"[Forecast] Warning: Failed to parse cached JSON response: {e}")
                    # Continue with fresh request
        
        # Track research data from tool calls
        research_data = []
        extracted_data = None
        
        # Check if we have a question analysis and it's a financial question 
        if hasattr(self, 'shared_context') and 'question_analysis' in self.shared_context:
            analysis = self.shared_context['question_analysis']
            
            # 1. Handle stock data for financial questions about stocks
            if analysis.get("question_type") == "financial" and analysis.get("entity") and hasattr(self, "_has_stock_data") and self._has_stock_data:
                # Try to determine if this is about a stock
                entity = analysis.get("entity", "").lower()
                crypto_keywords = ["crypto", "bitcoin", "ethereum", "coin", "token", "blockchain"]
                
                # If it doesn't contain crypto keywords, try stock data
                if not any(keyword in entity.lower() for keyword in crypto_keywords):
                    # Try to get stock data for the entity
                    ticker = analysis.get("identifier") or analysis.get("entity")
                    print(f"[Forecast] Getting stock data for {ticker}")
                    
                    try:
                        stock_tool = self.tools_registry.get("stock_data")
                        if stock_tool:
                            stock_data = stock_tool.run(ticker=ticker)
                            
                            if stock_data.get("success") and stock_data.get("price"):
                                # Use this as our extracted data
                                value = stock_data["price"]
                                unit = "dollars"
                                entity = stock_data.get("company_name", ticker)
                                
                                # Add to shared context
                                extracted_data = {
                                    "currentValue": value,
                                    "valueUnit": unit,
                                    "valueDate": stock_data.get("timestamp", "now"),
                                    "relevantInfo": f"The current price of {entity} ({ticker}) is ${value}."
                                }
                                
                                # Add change info if available
                                if "percent_change" in stock_data:
                                    extracted_data["changeValue"] = stock_data["percent_change"]
                                    extracted_data["changeUnit"] = "percent"
                                    extracted_data["changePeriod"] = "daily"
                                    extracted_data["relevantInfo"] += f" It has changed by {stock_data['percent_change']}% today."
                                
                                self.shared_context["extracted_data"] = extracted_data
                                self.shared_context["key_info"] = extracted_data["relevantInfo"]
                                
                                print(f"[Forecast] Added stock price data: ${value}")
                    except Exception as e:
                        print(f"[Forecast] Error getting stock data: {e}")
            
            # 2. Handle crypto data for questions about cryptocurrencies
            if analysis.get("question_type") == "financial" and analysis.get("entity") and hasattr(self, "_has_crypto_data") and self._has_crypto_data:
                # Try to determine if this is about a cryptocurrency
                entity = analysis.get("entity", "").lower()
                crypto_keywords = ["crypto", "bitcoin", "ethereum", "btc", "eth", "coin", "token", "blockchain"]
                crypto_entities = ["bitcoin", "ethereum", "ripple", "xrp", "cardano", "solana", "bnb", "doge"]
                
                # Check if this is likely a crypto query
                if any(keyword in entity.lower() for keyword in crypto_keywords) or entity.lower() in crypto_entities:
                    # Get crypto ID - use entity or identifier
                    crypto_id = analysis.get("identifier") or analysis.get("entity")
                    print(f"[Forecast] Getting crypto data for {crypto_id}")
                    
                    try:
                        crypto_tool = self.tools_registry.get("crypto_data")
                        if crypto_tool:
                            crypto_data = crypto_tool.run(crypto_id=crypto_id)
                            
                            if crypto_data.get("success") and crypto_data.get("price"):
                                # Use this as our extracted data
                                value = crypto_data["price"]
                                unit = crypto_data.get("price_unit", "USD")
                                name = crypto_data.get("name", crypto_id)
                                symbol = crypto_data.get("symbol", "").upper()
                                
                                # Add to shared context
                                extracted_data = {
                                    "currentValue": value,
                                    "valueUnit": unit,
                                    "valueDate": crypto_data.get("timestamp", "now"),
                                    "relevantInfo": f"The current price of {name} ({symbol}) is ${value}."
                                }
                                
                                # Add change info if available
                                if "percent_change_24h" in crypto_data:
                                    change = crypto_data["percent_change_24h"]
                                    extracted_data["changeValue"] = change
                                    extracted_data["changeUnit"] = "percent"
                                    extracted_data["changePeriod"] = "24h"
                                    extracted_data["relevantInfo"] += f" It has changed by {change}% in the last 24 hours."
                                
                                # Add market cap if available
                                if "market_cap" in crypto_data and crypto_data["market_cap"]:
                                    market_cap = crypto_data["market_cap"]
                                    if market_cap >= 1_000_000_000:
                                        cap_value = market_cap / 1_000_000_000
                                        cap_unit = "billion"
                                    else:
                                        cap_value = market_cap / 1_000_000
                                        cap_unit = "million"
                                    
                                    extracted_data["relevantInfo"] += f" Market cap is ${cap_value:.2f} {cap_unit}."
                                
                                self.shared_context["extracted_data"] = extracted_data
                                self.shared_context["key_info"] = extracted_data["relevantInfo"]
                                
                                print(f"[Forecast] Added crypto price data: ${value}")
                    except Exception as e:
                        print(f"[Forecast] Error getting crypto data: {e}")
        
        # First API call with tool access
        print("[Forecast] Generating initial forecast...")
        response = self._call_api_with_retries(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            response_format={"type": "json_object"} if not tools else None
        )
        
        # Handle tool calls if present
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        if tool_calls:
            # Save the original message that requested tools
            messages.append({"role": "assistant", "content": message.content, "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in tool_calls
            ]})
            
            # Process each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Use optimized query if available for web search
                if tool_name == "web_search" and 'optimized_query' in self.shared_context:
                    original_query = tool_args["query"]
                    tool_args["query"] = self.shared_context['optimized_query']
                    print(f"[Forecast] Using optimized query: '{self.shared_context['optimized_query']}' instead of '{original_query}'")
                
                # Call the tool
                result = self._call_tool(tool_name, tool_args)
                
                # Process web search results
                if tool_name == "web_search" and result.get("success"):
                    research_data = result["results"]
                    
                    # Extract key information from search results based on question type
                    if hasattr(self, 'shared_context') and 'question_analysis' in self.shared_context and not extracted_data:
                        analysis = self.shared_context['question_analysis']
                        extracted_data = self._extract_key_information(analysis, research_data)
                        
                        if extracted_data and extracted_data.get("relevantInfo"):
                            # Store this in shared context
                            self.shared_context["extracted_data"] = extracted_data
                            
                            # Build info context based on available data
                            key_info = f"\nIMPORTANT DATA: {extracted_data['relevantInfo']}"
                            
                            # Add specific value information if available
                            if "currentValue" in extracted_data and "valueUnit" in extracted_data:
                                value = extracted_data["currentValue"]
                                unit = extracted_data["valueUnit"]
                                date_info = ""
                                if "valueDate" in extracted_data:
                                    date_info = f" (as of {extracted_data['valueDate']})"
                                
                                key_info += f"\n\nCurrent value of {analysis.get('entity', 'the subject')}: {value} {unit}{date_info}"
                            
                            # Add change information if available
                            if "changeValue" in extracted_data and "changeUnit" in extracted_data:
                                change = extracted_data["changeValue"]
                                change_unit = extracted_data["changeUnit"]
                                period = extracted_data.get("changePeriod", "recent period")
                                
                                key_info += f"\nRecent change: {change} {change_unit} over the {period}"
                            
                            self.shared_context["key_info"] = key_info
                            print(f"[Forecast] Extracted relevant information from search results")
                
                # Process stock data results
                if tool_name == "stock_data" and result.get("success") and not extracted_data:
                    ticker = tool_args.get("ticker", "")
                    value = result.get("price")
                    unit = "dollars"
                    entity = result.get("company_name", ticker)
                    
                    if value:
                        # Create extracted data
                        extracted_data = {
                            "currentValue": value,
                            "valueUnit": unit,
                            "valueDate": result.get("timestamp", "now"),
                            "relevantInfo": f"The current price of {entity} ({ticker}) is ${value}."
                        }
                        
                        # Add change info if available
                        if "percent_change" in result:
                            extracted_data["changeValue"] = result["percent_change"]
                            extracted_data["changeUnit"] = "percent"
                            extracted_data["changePeriod"] = "daily"
                            extracted_data["relevantInfo"] += f" It has changed by {result['percent_change']}% today."
                        
                        self.shared_context["extracted_data"] = extracted_data
                        self.shared_context["key_info"] = extracted_data["relevantInfo"]
                        
                        print(f"[Forecast] Added stock price data: ${value}")
                
                # Process crypto data results
                if tool_name == "crypto_data" and result.get("success") and not extracted_data:
                    crypto_id = tool_args.get("crypto_id", "")
                    value = result.get("price")
                    unit = result.get("price_unit", "USD")
                    name = result.get("name", crypto_id)
                    symbol = result.get("symbol", "").upper()
                    
                    if value:
                        # Create extracted data
                        extracted_data = {
                            "currentValue": value,
                            "valueUnit": unit,
                            "valueDate": result.get("timestamp", "now"),
                            "relevantInfo": f"The current price of {name} ({symbol}) is ${value}."
                        }
                        
                        # Add change info if available
                        if "percent_change_24h" in result:
                            change = result["percent_change_24h"]
                            extracted_data["changeValue"] = change
                            extracted_data["changeUnit"] = "percent"
                            extracted_data["changePeriod"] = "24h"
                            extracted_data["relevantInfo"] += f" It has changed by {change}% in the last 24 hours."
                        
                        # Add market cap info if available
                        if "market_cap" in result and result["market_cap"]:
                            market_cap = result["market_cap"]
                            if market_cap >= 1_000_000_000:
                                cap_value = market_cap / 1_000_000_000
                                cap_unit = "billion"
                            else:
                                cap_value = market_cap / 1_000_000
                                cap_unit = "million"
                            
                            extracted_data["relevantInfo"] += f" Market cap is ${cap_value:.2f} {cap_unit}."
                        
                        self.shared_context["extracted_data"] = extracted_data
                        self.shared_context["key_info"] = extracted_data["relevantInfo"]
                        
                        print(f"[Forecast] Added crypto price data: ${value}")
                
                # Add the result back as a tool call response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
            # Add extracted info as a separate message if available
            if "key_info" in self.shared_context:
                messages.append({
                    "role": "user",
                    "content": self.shared_context["key_info"]
                })
            
            # Final API call to generate forecast
            print("[Forecast] Generating final forecast after research...")
            response = self._call_api_with_retries(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
        
        # Parse the final response
        content = response.choices[0].message.content
        
        try:
            result = json.loads(content)
            result["research"] = research_data
            
            # Cache the response if caching is enabled
            if not tool_calls:  # Only cache non-tool responses
                cache_key = self._get_cache_key(messages)
                self._save_to_cache(cache_key, response)
                
            return result
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            # Return a fallback result
            return {
                "forecast": "Unable to generate a forecast due to technical issues.",
                "probability": 0.5,
                "reasoning": f"Error parsing model response: {str(e)}",
                "research": research_data
            }
    
    def _run_consistency_check(self, question: str, check_type: str) -> Dict[str, Any]:
        """Run a consistency check of the specified type with tool support if needed."""
        # Create base message content
        if check_type == "temporal_decomposition":
            prompt = self._get_temporal_decomposition_prompt(question)
        elif check_type == "spatial_decomposition":
            prompt = self._get_spatial_decomposition_prompt(question)
        elif check_type == "chain_of_thought":
            prompt = self._get_chain_of_thought_prompt(question)
        elif check_type == "alternative_framing":
            prompt = self._get_alternative_framing_prompt(question)
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Add extracted value guidance if available
        if hasattr(self, 'shared_context') and 'extracted_data' in self.shared_context:
            extracted = self.shared_context['extracted_data']
            if "currentValue" in extracted and "valueUnit" in extracted:
                value = extracted["currentValue"]
                unit = extracted["valueUnit"]
                entity = self.shared_context['question_analysis'].get('entity', 'subject')
                
                guidance = f"\nIMPORTANT: Use {value} {unit} as the current value for {entity} in your analysis."
                messages.append({
                    "role": "user", 
                    "content": guidance
                })
                print(f"[Check {check_type}] Using value: {value} {unit}")
        
        # Set up tools (if available)
        tools = self.tools_registry.get_tool_descriptions() if self._has_web_search else None
        
        # Try to get cached response (only for non-tool requests)
        if not tools:
            cache_key = self._get_cache_key(messages)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                try:
                    result = json.loads(cached_response.choices[0].message.content)
                    result["type"] = check_type
                    return result
                except json.JSONDecodeError as e:
                    print(f"[Check {check_type}] Warning: Failed to parse cached JSON response: {e}")
                    # Continue with fresh request
        
        # Initial call
        print(f"[Check {check_type}] Running consistency check...")
        response = self._call_api_with_retries(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            response_format={"type": "json_object"} if not tools else None
        )
        
        # Handle tool calls if present
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        if tool_calls:
            # Save the original message
            messages.append({"role": "assistant", "content": message.content, "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in tool_calls
            ]})
            
            # Process each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Use optimized query if available
                if tool_name == "web_search" and 'optimized_query' in self.shared_context:
                    original_query = tool_args["query"]
                    tool_args["query"] = self.shared_context['optimized_query']
                    print(f"[Check {check_type}] Using optimized query: '{self.shared_context['optimized_query']}' instead of '{original_query}'")
                
                # Call the tool
                result = self._call_tool(tool_name, tool_args)
                
                # Add extracted information if this is a web search
                if tool_name == "web_search" and result.get("success"):
                    research_data = result["result"]["results"]
                    
                    # Extract key information from search results
                    if hasattr(self, 'shared_context') and 'question_analysis' in self.shared_context:
                        analysis = self.shared_context['question_analysis']
                        extracted_data = self._extract_key_information(analysis, research_data)
                        
                        if extracted_data and extracted_data.get("relevantInfo"):
                            # Build info context based on available data
                            key_info = f"\nIMPORTANT DATA: {extracted_data['relevantInfo']}"
                            
                            # Add specific value information if available
                            if "currentValue" in extracted_data and "valueUnit" in extracted_data:
                                value = extracted_data["currentValue"]
                                unit = extracted_data["valueUnit"]
                                date_info = ""
                                if "valueDate" in extracted_data:
                                    date_info = f" (as of {extracted_data['valueDate']})"
                                
                                key_info += f"\n\nCurrent value of {analysis.get('entity', 'the subject')}: {value} {unit}{date_info}"
                            
                            # Add change information if available
                            if "changeValue" in extracted_data and "changeUnit" in extracted_data:
                                change = extracted_data["changeValue"]
                                change_unit = extracted_data["changeUnit"]
                                period = extracted_data.get("changePeriod", "recent period")
                                
                                key_info += f"\nRecent change: {change} {change_unit} over the {period}"
                                
                            key_name = f"key_info_{check_type}"
                            self.shared_context[key_name] = key_info
                            print(f"[Check {check_type}] Added key information context from search results")
                
                # Add the tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })
            
            # After all tool calls are processed, add any extracted information
            key_name = f"key_info_{check_type}"
            if key_name in self.shared_context:
                messages.append({
                    "role": "user",
                    "content": self.shared_context[key_name]
                })
                # Clean up after using
                del self.shared_context[key_name]
            
            # Final call after tool usage
            print(f"[Check {check_type}] Finalizing consistency check...")
            response = self._call_api_with_retries(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
        elif not tools:
            # Cache the response if it's a non-tool request
            self._save_to_cache(cache_key, response)
        
        # Parse and return result
        try:
            result = json.loads(response.choices[0].message.content)
            result["type"] = check_type
        except json.JSONDecodeError as e:
            print(f"[Check {check_type}] Error parsing JSON response: {e}")
            print(f"[Check {check_type}] Raw content: {response.choices[0].message.content}")
            # Return a fallback response
            result = {
                "type": check_type,
                "reasoning": "Error parsing model response. The model did not return valid JSON.",
                "probability": 0.5
            }
            # Add specific fields based on check type
            if check_type in ["temporal_decomposition", "spatial_decomposition"]:
                result["decomposition"] = "Error in decomposition generation"
                if check_type == "temporal_decomposition":
                    result["key_uncertainties"] = "Error in uncertainty analysis"
                else:
                    result["key_dependencies"] = "Error in dependency analysis"
            elif check_type == "alternative_framing":
                result["reframing"] = "Error in reframing generation"
        
        return result
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        system_prompt = """
        You are an AI forecasting assistant helping make accurate predictions about future events.
        
        Your task is to provide detailed, step-by-step probability estimates for questions about future events.
        
        Before making predictions, consider using available tools to gather real-time information 
        that you might not have up-to-date knowledge about.
        
        When making your forecast:
        1. Provide detailed step-by-step reasoning with clear logic chains
        2. Synthesize information from your knowledge and research
        3. Express uncertainty appropriately and explain its sources
        4. Consider alternative scenarios
        5. Explicitly discuss key assumptions
        6. Return a numeric probability between 0 and 1
        """
        return system_prompt
    
    def _get_user_prompt(self, question: str) -> str:
        """Get the user prompt for the LLM."""
        if self._has_web_search:
            return f"""
            Please forecast the following question: {question}
            
            Feel free to use the web_search tool if you need current information or data.
            
            In your reasoning, please be extremely explicit about:
            1. What information you're using from your searches
            2. How you're interpreting that information
            3. Key assumptions you're making
            4. Alternative scenarios you've considered
            5. Sources of uncertainty in your estimate
            
            Return your response as a JSON object with the following format:
            {{
                "reasoning": "your detailed step-by-step thinking including assumptions and alternative scenarios",
                "probability": "IMPORTANT: Just the numeric value between 0 and 1 (e.g., 0.7)",
                "forecast": "your one-sentence forecast",
                "key_factors": "list the 3-5 most important factors influencing your forecast",
                "confidence_explanation": "explain why you are more or less confident in this forecast"
            }}
            """
        else:
            return f"""
            Please forecast the following question: {question}
            
            In your reasoning, please be extremely explicit about:
            1. Key assumptions you're making
            2. Alternative scenarios you've considered
            3. Sources of uncertainty in your estimate
            
            Return your response as a JSON object with the following format:
            {{
                "reasoning": "your detailed step-by-step thinking including assumptions and alternative scenarios",
                "probability": "IMPORTANT: Just the numeric value between 0 and 1 (e.g., 0.7)",
                "forecast": "your one-sentence forecast",
                "key_factors": "list the 3-5 most important factors influencing your forecast",
                "confidence_explanation": "explain why you are more or less confident in this forecast"
            }}
            """
    
    def _get_temporal_decomposition_prompt(self, question: str) -> str:
        """Create a prompt for temporal decomposition"""
        return f"""
        Consider the following question: {question}
        
        Break this question down into a sequence of events over time. Then estimate the probability of each event, 
        and use them to calculate the overall probability.
        
        In your response:
        1. Clearly identify each time period or sequential event
        2. Explain your reasoning for each probability estimate
        3. Show your mathematical work for combining these probabilities
        4. Discuss how uncertainty compounds or resolves over time
        
        IMPORTANT: Always report the probability for the exact question as stated above, not its negation.
        If you calculate the probability of the opposite outcome, convert it to the original question's probability.
        For price-related questions, always use the most current price data available.
        
        Return your response as a JSON object with the following format:
        {{
            "decomposition": "your detailed temporal breakdown with specific time periods",
            "reasoning": "your step-by-step probability calculation with clear mathematical justification",
            "probability": "IMPORTANT: Just the numeric value between 0 and 1 (e.g., 0.7) for the original question",
            "key_uncertainties": "list the main sources of uncertainty in your temporal model"
        }}
        """
    
    def _get_spatial_decomposition_prompt(self, question: str) -> str:
        """Create a prompt for spatial/component decomposition"""
        return f"""
        Consider the following question: {question}
        
        Break this question down into its key components or factors. Estimate the probability for each component, 
        and then combine them to calculate the overall probability.
        
        In your response:
        1. Clearly identify each component or factor
        2. Explain your reasoning for each component's probability
        3. Show your mathematical work for combining these probabilities
        4. Discuss any dependencies between components
        
        IMPORTANT: Always report the probability for the exact question as stated above, not its negation.
        If you calculate the probability of the opposite outcome, convert it to the original question's probability.
        For price-related questions, always use the most current price data available.
        
        Return your response as a JSON object with the following format:
        {{
            "decomposition": "your detailed component breakdown with specific factors",
            "reasoning": "your step-by-step probability calculation with clear mathematical justification",
            "probability": "IMPORTANT: Just the numeric value between 0 and 1 (e.g., 0.7) for the original question",
            "key_dependencies": "explain any important dependencies between components in your model"
        }}
        """
    
    def _get_chain_of_thought_prompt(self, question: str) -> str:
        """Create a prompt for chain-of-thought reasoning"""
        return f"""
        Consider the following question: {question}
        
        Think through this question using a different chain of reasoning than you might normally use.
        Consider alternative perspectives and approach the problem in a novel way.
        
        IMPORTANT: Always report the probability for the exact question as stated above, not its negation.
        If you calculate the probability of the opposite outcome, convert it to the original question's probability.
        For price-related questions, always use the most current price data available.
        
        Return your response as a JSON object with the following format:
        {{
            "reasoning": "your alternative chain-of-thought",
            "probability": "IMPORTANT: Just the numeric value between 0 and 1 (e.g., 0.7) for the original question"
        }}
        """
    
    def _get_alternative_framing_prompt(self, question: str) -> str:
        """Create a prompt for alternative framing"""
        return f"""
        Consider the following question: {question}
        
        Reframe this question in a different way that might lead to a different perspective.
        Then answer the reframed question and see if you get a similar answer to the original.
        
        IMPORTANT: While you may reframe the question in your analysis, always report the final probability 
        for the exact original question as stated above, not its negation or your reframed version.
        If you calculate the probability of the opposite outcome, convert it to the original question's probability.
        For price-related questions, always use the most current price data available.
        
        Return your response as a JSON object with the following format:
        {{
            "reframing": "your alternative framing of the question",
            "reasoning": "your analysis of the reframed question",
            "probability": "IMPORTANT: Just the numeric value between 0 and 1 (e.g., 0.7) for the original question"
        }}
        """
    
    def _calculate_consistency_score(self, probabilities: List[float]) -> float:
        """
        Calculate a consistency score based on the variability of probability estimates
        
        Lower variance = higher consistency
        """
        if not probabilities:
            return 0.0
            
        # Calculate variance
        mean = sum(probabilities) / len(probabilities)
        variance = sum((p - mean) ** 2 for p in probabilities) / len(probabilities)
        
        # Convert variance to a consistency score (1 = perfectly consistent, 0 = completely inconsistent)
        # Using a simple transformation: consistency = 1 - min(1, 4*variance)
        # This makes small variances (< 0.25) map to higher consistency scores
        consistency = 1 - min(1, 4 * variance)
        
        return round(consistency, 2)
    
    def generate_backtest_forecast(self, question: str, cutoff_date: str, num_checks: int = 3, verbose: bool = False, progress_callback=None) -> Dict[str, Any]:
        """
        Generate a forecast for backtesting with a specified historical cutoff date.
        
        Args:
            question: The forecasting question
            cutoff_date: Date in YYYY-MM-DD format representing the knowledge cutoff
            num_checks: Number of consistency checks to perform
            verbose: Whether to include detailed reasoning
            progress_callback: Optional callback function to track progress
            
        Returns:
            Dictionary containing the forecast and consistency data
        """
        print(f"[Backtest] Running backtest with cutoff date: {cutoff_date}")
        
        # Store original tools and settings
        original_tools_registry = self.tools_registry
        original_get_system_prompt = self._get_system_prompt
        original_get_user_prompt = self._get_user_prompt

        try:
            # Create time-bounded tools
            backtest_tools = BacktestToolRegistry(cutoff_date=cutoff_date)
            self.tools_registry = backtest_tools
            
            # Add time constraint to shared context
            if not hasattr(self, 'shared_context'):
                self.shared_context = {}
            self.shared_context['backtest_mode'] = True
            self.shared_context['cutoff_date'] = cutoff_date
            
            # Override system prompt method with time-bounded version
            self._get_system_prompt = lambda: self._get_system_prompt_with_time_bound(cutoff_date)
            
            # Override user prompt method to include date information
            # Don't use lambda here to avoid recursion
            def time_bounded_user_prompt(q):
                base_prompt = original_get_user_prompt(q)
                time_bound_note = f"""
                IMPORTANT: You are making this forecast on {cutoff_date}. Only use information that would have been
                available on or before this date. Ignore any knowledge you have about events after {cutoff_date}.
                """
                return base_prompt + time_bound_note
            
            self._get_user_prompt = time_bounded_user_prompt
            
            # Run the forecast with modified context
            print(f"[Backtest] Generating forecast for: '{question}' as if we're forecasting from {cutoff_date}")
            result = self.generate_forecast(question, num_checks=num_checks, verbose=verbose, progress_callback=progress_callback)
            
            # Add backtest metadata
            result["backtest_metadata"] = {
                "cutoff_date": cutoff_date,
                "is_backtest": True,
                "mode": "time_bounded"
            }
            
            return result
            
        finally:
            # Always restore original methods and tools
            self.tools_registry = original_tools_registry
            self._get_system_prompt = original_get_system_prompt
            self._get_user_prompt = original_get_user_prompt
            
            # Clean up shared context
            if hasattr(self, 'shared_context'):
                if 'backtest_mode' in self.shared_context:
                    del self.shared_context['backtest_mode']
                if 'cutoff_date' in self.shared_context:
                    del self.shared_context['cutoff_date']
    
    def _get_system_prompt_with_time_bound(self, cutoff_date: str) -> str:
        """Get the system prompt with time boundary for backtesting."""
        return f"""
        You are an AI forecasting assistant helping make accurate predictions about future events.
        
        IMPORTANT: You must ONLY use knowledge available prior to {cutoff_date}. Do NOT use any information 
        about events that occurred after this date. Pretend you don't know what happened after this date.
        
        Your task is to provide detailed, step-by-step probability estimates for questions about future events.
        
        Before making predictions, consider using available tools to gather real-time information 
        that you might not have up-to-date knowledge about, but remember to ONLY use information
        available before {cutoff_date}.
        
        When making your forecast:
        1. Provide detailed step-by-step reasoning with clear logic chains
        2. Synthesize information from your knowledge and research
        3. Express uncertainty appropriately and explain its sources
        4. Consider alternative scenarios
        5. Explicitly discuss key assumptions
        6. Return a numeric probability between 0 and 1
        """
    
    def _extract_key_information(self, analysis: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generic method to extract key information from search results based on question type.
        Uses LLM to intelligently extract the most relevant data points.
        """
        if not search_results:
            return None
            
        # Extract question type and details
        question_type = analysis.get("question_type", "")
        entity = analysis.get("entity", "")
        identifier = analysis.get("identifier", "")
        threshold = analysis.get("threshold", "")
        threshold_unit = analysis.get("threshold_unit", "")
        timeframe = analysis.get("timeframe", "")
        
        # Create a simplified version of the search results
        result_texts = []
        for idx, result in enumerate(search_results[:5]):  # Limit to first 5 results
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            source = result.get("link", "")
            
            result_texts.append(f"Result {idx+1}:\nTitle: {title}\nSnippet: {snippet}\nSource: {source}\n")
            
        results_text = "\n".join(result_texts)
        
        # Create a prompt that guides the LLM to extract relevant information for any domain
        prompt = f"""
        You are an expert information extractor that identifies key factual data from search results.
        
        Information about the question being forecasted:
        - Domain/Type: {question_type}
        - Main entity/subject: {entity}
        """
        
        # Add additional context based on what's available
        if identifier:
            prompt += f"- Identifier (e.g., ticker symbol, code): {identifier}\n"
        if threshold and threshold_unit:
            prompt += f"- Related value of interest: {threshold} {threshold_unit}\n"
        if timeframe:
            prompt += f"- Time frame: {timeframe}\n"
            
        prompt += f"""
        Here are the search results:
        
        {results_text}
        
        Your task is to extract the most important factual information that would help answer questions about {entity}.
        
        Focus on extracting:
        1. The most current/recent value, price, or measurement related to {entity}
        2. Recent changes, trends, or movements (including percentages or absolute changes)
        3. Important contextual information about current status or conditions
        4. Any time-sensitive information that affects forecasting
        5. Information from the most reliable sources in the results
        
        For all numeric values, include the exact numbers found in the text.
        
        Return your answer as a JSON object with these fields:
        - relevantInfo: A concise paragraph summarizing the most important factual information (1-3 sentences)
        - currentValue: If available, the current value, price, or measurement related to the question (as a number)
        - valueUnit: The unit of measurement for the currentValue (e.g., "dollars", "percent", "points")
        - valueDate: The date associated with the currentValue, if available
        - changeValue: Any recent change or movement value (as a number)
        - changeUnit: The unit for the change (e.g., "percent", "points", "dollars")
        - changePeriod: The time period for the change (e.g., "day", "week", "month")
        - confidence: Your confidence in this information (high, medium, or low)
        - source: The URL source of the most reliable information
        
        Only include fields where you found relevant information. Omit fields where no information was found.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                extracted_data = json.loads(content)
                
                if "relevantInfo" in extracted_data:
                    print(f"[Forecast] Extracted relevant information from search results")
                    
                    # Print extracted value if available
                    if "currentValue" in extracted_data:
                        unit = extracted_data.get("valueUnit", "")
                        conf = extracted_data.get("confidence", "unknown")
                        print(f"[Forecast] Found current value: {extracted_data['currentValue']} {unit} (confidence: {conf})")
                    
                    return extracted_data
                    
            except json.JSONDecodeError:
                print(f"[Forecast] Error parsing JSON in key information extraction")
                
            return None
            
        except Exception as e:
            print(f"[Forecast] Error extracting key information from search results: {e}")
            return None 