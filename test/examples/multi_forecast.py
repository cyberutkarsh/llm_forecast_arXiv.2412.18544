#!/usr/bin/env python3
"""
Multi-asset forecasting script that generates forecasts for multiple stocks or cryptocurrencies
and sorts them by likelihood.

Usage:
    python multi_forecast.py --assets AAPL,MSFT,GOOGL --type stock
    python multi_forecast.py --assets BTC,ETH,SOL --type crypto
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Add the parent directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Now imports should work correctly
from src.forecaster import ForecastingSystem
from utils.token_utils import TokenTracker, MODEL_PRICING

# OpenAI API pricing information (per 1M tokens as of May 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00, "name": "GPT-4o"},
    "o3": {"input": 5.00, "output": 15.00, "name": "GPT-4o"},  # Alias
    "gpt-4o-mini": {"input": 0.50, "output": 1.50, "name": "GPT-4o-mini"},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "name": "GPT-4 Turbo"},
    "gpt-4": {"input": 30.00, "output": 60.00, "name": "GPT-4"},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "name": "GPT-3.5 Turbo"},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate forecasts for multiple assets")
    parser.add_argument("--assets", type=str, required=True, 
                      help="Comma-separated list of asset symbols (e.g., AAPL,MSFT,GOOGL or BTC,ETH,SOL)")
    parser.add_argument("--type", type=str, choices=["stock", "crypto"], required=True,
                      help="Type of assets: stock or crypto")
    parser.add_argument("--timeframe", type=str, default="next month", 
                      help="Timeframe for forecast (e.g., 'next week', 'next month', 'by end of year')")
    parser.add_argument("--direction", type=str, default="increase", choices=["increase", "decrease", "change"],
                      help="Direction of forecast")
    parser.add_argument("--num_checks", type=int, default=3,
                      help="Number of consistency checks to run")
    parser.add_argument("--model", type=str, default="o3",
                      help="Model to use for forecasting")
    parser.add_argument("--verbose", action="store_true",
                      help="Show verbose output")
    return parser.parse_args()

def get_forecast_question(asset: str, asset_type: str, timeframe: str, direction: str) -> str:
    """Generate a forecasting question for the given asset."""
    
    if asset_type == "stock":
        return f"Will {asset} stock price {direction} in the {timeframe}?"
    else:  # crypto
        return f"Will the price of {asset} {direction} in the {timeframe}?"

def generate_forecasts(assets: List[str], asset_type: str, timeframe: str, 
                       direction: str, num_checks: int, model: str, verbose: bool) -> Dict[str, Any]:
    """Generate forecasts for multiple assets."""
    
    results = []
    
    # Initialize token tracker for the specific model
    token_tracker = TokenTracker(model)
    
    # Initialize forecasting system
    print("Initializing forecasting system...")
    forecaster = ForecastingSystem(
        model=model,
        analyzer_model="gpt-4o",
        cache_dir=".forecast_cache",
        max_retries=3,
        max_workers=4,
        disable_cache=False
    )
    
    # Monitor LLM API calls for token usage using our monkey patch method
    # This is a fallback in case the forecaster doesn't have built-in token tracking
    original_llm_call = None
    if hasattr(forecaster, "_call_api_with_retries"):
        original_llm_call = forecaster._call_api_with_retries
        
        def patched_llm_call(*args, **kwargs):
            """Monkey patched LLM call function to track tokens."""
            response = original_llm_call(*args, **kwargs)
            
            # Try to extract token counts from the response
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                token_tracker.add_tokens(
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0)
                )
            
            return response
        
        # Apply the monkey patch
        forecaster._call_api_with_retries = patched_llm_call
        print("[Setup] Token tracking enabled")
    
    # Enable/disable appropriate tools based on asset type
    if asset_type == "stock":
        if "crypto_data" in forecaster.tools_registry:
            forecaster.tools_registry.deregister("crypto_data")
            forecaster._has_crypto_data = False
    else:  # crypto
        if "stock_data" in forecaster.tools_registry:
            forecaster.tools_registry.deregister("stock_data")
            forecaster._has_stock_data = False
    
    # Generate forecasts for each asset
    print(f"\nGenerating forecasts for {len(assets)} {asset_type}s...")
    for asset in tqdm(assets, desc=f"Forecasting {asset_type}s"):
        question = get_forecast_question(asset, asset_type, timeframe, direction)
        print(f"\nProcessing: {question}")
        
        try:
            # Generate forecast
            start_time = time.time()
            result = forecaster.generate_forecast(
                question=question,
                num_checks=num_checks,
                verbose=verbose
            )
            execution_time = time.time() - start_time
            
            # Add metadata to result
            result["asset"] = asset
            result["asset_type"] = asset_type
            result["question"] = question
            result["execution_time"] = execution_time
            
            results.append(result)
            print(f"✓ Completed forecast for {asset} in {execution_time:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error generating forecast for {asset}: {str(e)}")
    
    # Restore original LLM call method if we patched it
    if original_llm_call:
        forecaster._call_api_with_retries = original_llm_call
    
    # Return results with token usage data
    return {
        "forecasts": results,
        "token_tracker": token_tracker,
        "model": model
    }

def calculate_api_cost(token_usage: Dict[str, int], model: str) -> Dict[str, float]:
    """Calculate the estimated OpenAI API cost based on token usage and model."""
    
    # Get pricing for the model
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING.get("o3"))  # Default to o3/gpt-4o if not found
    
    # Calculate costs
    input_cost = (token_usage.get("input", 0) / 1_000_000) * pricing["input"]
    output_cost = (token_usage.get("output", 0) / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model_name": pricing["name"],
        "input_price_per_1m": pricing["input"],
        "output_price_per_1m": pricing["output"]
    }

def sort_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Sort results by confidence (high to low)."""
    
    forecasts = results["forecasts"]
    
    def get_sort_key(result):
        # Extract confidence value, converting to float if needed
        confidence = result.get("confidence", 0)
        if isinstance(confidence, str):
            # Try to extract numeric part if it's a string
            import re
            matches = re.findall(r'(\d+(\.\d+)?)', confidence)
            if matches:
                try:
                    return float(matches[0][0]) / 100  # Convert percentage to decimal
                except ValueError:
                    return 0
            return 0
        return confidence
    
    # Sort forecasts by confidence (high to low)
    sorted_forecasts = sorted(forecasts, key=get_sort_key, reverse=True)
    
    # Return a new results dict with sorted forecasts
    return {
        "forecasts": sorted_forecasts,
        "token_tracker": results["token_tracker"],
        "model": results["model"]
    }

def display_results(results: Dict[str, Any]):
    """Display the sorted forecast results."""
    
    forecasts = results["forecasts"]
    token_tracker = results["token_tracker"]
    
    print("\n" + "="*80)
    print(f"FORECAST RESULTS (SORTED BY CONFIDENCE)")
    print("="*80)
    
    for i, result in enumerate(forecasts, 1):
        asset = result.get("asset", "Unknown")
        asset_type = result.get("asset_type", "Unknown")
        question = result.get("question", "Unknown")
        forecast = result.get("forecast", "No forecast")
        confidence = result.get("confidence", "Unknown")
        
        # Get current price if available
        current_price = None
        if "question_analysis" in result and result["question_analysis"]:
            # Try to find price in research data
            if "research" in result and result["research"]:
                for item in result["research"]:
                    if "extracted_information" in item and "values" in item["extracted_information"]:
                        for value in item["extracted_information"]["values"]:
                            if value.get("entity", "").lower() == asset.lower() and "price" in value.get("description", "").lower():
                                current_price = f"{value.get('value')} {value.get('unit', '')}"
                                break
        
        # Print summary
        print(f"\n{i}. {asset} ({asset_type.upper()})")
        print(f"   Question: {question}")
        if current_price:
            print(f"   Current price: {current_price}")
        print(f"   Forecast: {forecast}")
        print(f"   Confidence: {confidence}")
        print(f"   Time: {result.get('execution_time', 0):.2f} seconds")
        
        # If there are key factors, display them
        if "key_factors" in result:
            print(f"\n   Key factors:")
            print(f"   {result['key_factors']}")
        
        print("-"*80)
    
    # Display API usage and cost summary
    token_tracker.print_summary()

def main():
    """Main entry point for the script."""
    
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Process asset list
    assets = [asset.strip() for asset in args.assets.split(",")]
    
    # Generate forecasts
    results = generate_forecasts(
        assets=assets,
        asset_type=args.type,
        timeframe=args.timeframe,
        direction=args.direction,
        num_checks=args.num_checks,
        model=args.model,
        verbose=args.verbose
    )
    
    # Sort results by confidence
    sorted_results = sort_results(results)
    
    # Display results
    display_results(sorted_results)
    
    print(f"\nCompleted forecasts for {len(results['forecasts'])} assets.")
    
    return sorted_results

if __name__ == "__main__":
    main() 