#!/usr/bin/env python3
import os
import argparse
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
from forecaster import ForecastingSystem
from explainer import ForecastExplainer

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Forecasting with Consistency Checks")
    parser.add_argument("--question", type=str, required=True, help="The question to forecast")
    parser.add_argument("--model", type=str, default="o3", help="The LLM model to use")
    parser.add_argument("--num_checks", type=int, default=3, help="Number of consistency checks")
    parser.add_argument("--verbose", action="store_true", help="Show detailed chain of thought")
    parser.add_argument("--disable_web_search", action="store_true", help="Disable web search tool")
    parser.add_argument("--disable_stock_data", action="store_true", help="Disable stock data tool")
    parser.add_argument("--disable_crypto_data", action="store_true", help="Disable cryptocurrency data tool")
    parser.add_argument("--disable_cache", action="store_true", help="Disable response caching")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of API call retries")
    parser.add_argument("--cache_dir", type=str, default=".forecast_cache", help="Directory for caching")
    
    # Add backtesting options
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--cutoff_date", type=str, help="Knowledge cutoff date for backtesting (YYYY-MM-DD)")
    
    return parser.parse_args()

def check_api_keys():
    """Check which API keys are available and provide instructions if missing."""
    missing_keys = []
    
    # Check OpenAI API Key (required)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("This is required for all functionality")
        print("  - Sign up at https://platform.openai.com/")
        print("  - Create an API key at https://platform.openai.com/api-keys")
        print("  - Add to .env file: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Check Google API keys (for web search)
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CX"):
        missing_keys.append("Web search")
        print("NOTE: Google API keys (GOOGLE_API_KEY and GOOGLE_CX) not found")
        print("  Web search functionality will be disabled")
        print("  To enable:")
        print("  - Create a Google Cloud account and enable Custom Search API")
        print("  - Create API key at https://console.cloud.google.com/apis/credentials")
        print("  - Set up a Custom Search Engine at https://programmablesearchengine.google.com/")
        print("  - Add to .env file: GOOGLE_API_KEY=your_key_here")
        print("  - Add to .env file: GOOGLE_CX=your_cx_here")
    
    # Check Alpha Vantage API key (for stock data)
    if not os.getenv("ALPHA_VANTAGE_API_KEY"):
        missing_keys.append("Stock data")
        print("NOTE: Alpha Vantage API key (ALPHA_VANTAGE_API_KEY) not found")
        print("  Stock data functionality will be disabled")
        print("  To enable:")
        print("  - Sign up for a free API key at https://www.alphavantage.co/support/#api-key")
        print("  - Add to .env file: ALPHA_VANTAGE_API_KEY=your_key_here")
    
    return missing_keys

def main():
    """Main entry point for the forecasting tool."""
    # Load environment variables
    load_dotenv()
    
    # Check API keys and provide guidance
    print("\nChecking API keys...\n")
    missing_keys = check_api_keys()
    print("")
    
    # Parse command line arguments
    args = parse_args()
    
    # Start timer
    start_time = time.time()
    
    print("Initializing forecasting system...")
    
    # Set up forecasting system with proper cache handling
    forecaster = ForecastingSystem(
        model=args.model,
        cache_dir=args.cache_dir,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        analyzer_model="gpt-4o",
        disable_cache=args.disable_cache
    )
    
    # Handle disabled tools
    tools_to_disable = []
    if args.disable_web_search or "Web search" in missing_keys:
        tools_to_disable.append("web_search")
    
    if args.disable_stock_data or "Stock data" in missing_keys:
        tools_to_disable.append("stock_data")
        
    if args.disable_crypto_data:
        tools_to_disable.append("crypto_data")
    
    # Deregister the disabled tools
    for tool_name in tools_to_disable:
        if tool_name in forecaster.tools_registry:
            forecaster.tools_registry.deregister(tool_name)
            print(f"[Setup] Disabled {tool_name} tool")
    
    # Update the feature flags
    forecaster._has_web_search = "web_search" in forecaster.tools_registry
    forecaster._has_stock_data = "stock_data" in forecaster.tools_registry
    forecaster._has_crypto_data = "crypto_data" in forecaster.tools_registry
    
    print(f"\nGenerating forecast for: '{args.question}'")
    print("Running initial forecast...")
    
    # Show progress
    with tqdm(total=args.num_checks+1, desc="Progress") as pbar:
        
        # Update progress callback
        def update_progress():
            pbar.update(1)
        
        # Run either backtest or regular forecast
        if args.backtest and args.cutoff_date:
            result = forecaster.generate_backtest_forecast(
                args.question, 
                args.cutoff_date,
                num_checks=args.num_checks,
                verbose=args.verbose,
                progress_callback=update_progress
            )
        else:
            result = forecaster.generate_forecast(
                args.question, 
                num_checks=args.num_checks,
                verbose=args.verbose,
                progress_callback=update_progress
            )
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    # Display results
    display_results(args.question, result, args.verbose, execution_time)
    
    # Return result for potential further use
    return result

def display_results(question, result, verbose, execution_time):
    print("\n" + "="*50)
    print(f"Question: {question}")
    if "backtest_metadata" in result and result["backtest_metadata"] and result["backtest_metadata"].get("cutoff_date"):
        print(f"Backtest Mode - Cutoff Date: {result['backtest_metadata']['cutoff_date']}")
    print("="*50)
    
    # Display question analysis if available
    if "question_analysis" in result and result["question_analysis"]:
        analysis = result["question_analysis"]
        print("\nQuestion Analysis:")
        print(f"  Type: {analysis.get('question_type', 'General')}")
        
        # Display key fields from analysis
        for key, value in analysis.items():
            if key not in ["analysis_success", "question_type"] and value is not None:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("="*50)
    
    print(f"Final forecast: {result['forecast']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # Display key factors if available
    if "key_factors" in result:
        print("\nKey factors affecting this forecast:")
        print(result["key_factors"])
    
    # Display confidence explanation if available  
    if "confidence_explanation" in result:
        print("\nConfidence explanation:")
        print(result["confidence_explanation"])
    
    # Show research results if available
    if "research" in result and result["research"]:
        print("\nResearch findings:")
        for i, item in enumerate(result["research"], 1):
            print(f"\n[{i}] {item.get('title', 'No title')}")
            print(f"    Link: {item.get('link', 'No link')}")
            if "snippet" in item:
                print(f"    Snippet: {item.get('snippet')}")
    
    if verbose:
        # Show initial reasoning
        if "reasoning" in result:
            print("\nReasoning:")
            print(result.get("reasoning", "No reasoning provided"))
            
        print("\nConsistency Check Results:")
        for i, check in enumerate(result['consistency_checks']):
            print(f"\nCheck {i+1}: {check['type']}")
            # Add explanation of check type
            print(f"  What this check does: {ForecastExplainer().explain_check_type(check['type'])[:150]}...")
            print(f"  Probability: {check['probability']}")
            if "reasoning" in check:
                print(f"  Reasoning: {check['reasoning']}")
            elif "decomposition" in check:
                print(f"  Decomposition: {check['decomposition']}")
                
            # Show additional check details if available
            if "key_uncertainties" in check:
                print(f"  Key uncertainties: {check['key_uncertainties']}")
            if "key_dependencies" in check:
                print(f"  Key dependencies: {check['key_dependencies']}")
        
        # Get probability values - make sure they're floats
        # Convert probability values from strings to float if necessary
        try:
            initial_prob = float(result['initial_probability']) if isinstance(result['initial_probability'], str) else result['initial_probability']
            check_probs = []
            
            for check in result['consistency_checks']:
                prob = check['probability']
                # Convert to float if it's a string
                if isinstance(prob, str):
                    try:
                        prob = float(prob)
                    except ValueError:
                        # If it can't be directly converted, try to extract a number
                        import re
                        matches = re.findall(r'0\.\d+|\d+\.\d+|\d+', prob)
                        if matches:
                            prob = float(matches[0])
                        else:
                            prob = 0.5  # Default if no number found
                check_probs.append(prob)
                
            probabilities = [initial_prob] + check_probs
            check_types = ["initial_forecast"] + [check['type'] for check in result['consistency_checks']]
            
            # Add detailed consistency explanation
            explanation = ForecastExplainer().explain_consistency_score(probabilities)
            print("\nConsistency Analysis:")
            print(f"- Mean probability: {explanation['mean_probability']}")
            print(f"- Standard deviation: {explanation['standard_deviation']}")
            print(f"- Range (max - min): {explanation['range']}")
            print(f"- Consistency score: {explanation['consistency_score']}")
            print(f"\nInterpretation: {explanation['interpretation']}")
            
            # Add visualization
            print(ForecastExplainer().generate_visual_explanation(probabilities))
            
            # Explain divergence if consistency is not high
            if result['consistency_score'] < 0.85:
                print("\nExplaining the divergence in probability estimates:")
                print(ForecastExplainer().explain_divergence(probabilities, check_types))
            
            # Add explanation of consistency score calculation
            print("\nOverall consistency score:", result['consistency_score'])
            print("\nHow the consistency score was calculated:")
            print("1. The system compares probability estimates across all checks")
            print("2. It measures the variance between these estimates")
            print("3. Low variance (similar estimates) produces a high consistency score")
            print("4. High variance (divergent estimates) produces a low consistency score")
            
            # Add interpretation guidance
            if result['consistency_score'] > 0.85:
                print("\nHigh consistency suggests strong agreement between different reasoning methods.")
                print("This typically indicates higher confidence in the forecast.")
            elif result['consistency_score'] > 0.7:
                print("\nMedium consistency indicates moderate agreement between reasoning methods.")
                print("Some divergence exists in how different approaches assess this question.")
            else:
                print("\nLow consistency indicates significant disagreement between reasoning methods.")
                print("This suggests high uncertainty and that the forecast should be treated with caution.")
        except Exception as e:
            print(f"\nError calculating consistency metrics: {str(e)}")
            print("The system was unable to properly analyze the consistency of probability estimates.")
        
    # Show optimization stats if available
    if "optimization_stats" in result:
        print("\nOptimization Statistics:")
        stats = result["optimization_stats"]
        if "cache_hits" in stats:
            print(f"- Cache hits: {stats['cache_hits']}")
        if "api_calls" in stats:
            print(f"- API calls: {stats['api_calls']}")
        if "parallel_time_saved" in stats:
            print(f"- Time saved by parallelization: {stats['parallel_time_saved']:.2f} seconds")
    
    # Show backtest metadata if available
    if "backtest_metadata" in result and result["backtest_metadata"]:
        print("\nBacktest Information:")
        print(f"- Cutoff Date: {result['backtest_metadata']['cutoff_date']}")
        print("- Note: The forecast above was generated as if it were made on this date.")
        print("  Any knowledge about events after this date was intentionally excluded.")

if __name__ == "__main__":
    main() 