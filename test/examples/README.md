# Multi-Asset Forecasting Tool

This tool allows you to generate forecasts for multiple stocks or cryptocurrencies and sorts the results by confidence level.

## Prerequisites

Ensure you have set up the required API keys in your `.env` file:

```
OPENAI_API_KEY=your_openai_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key  # Required for stock forecasts
GOOGLE_API_KEY=your_google_key                # For web search fallback
GOOGLE_CX=your_google_cx                      # For web search fallback
```

Also install the required packages:
```bash
pip install tiktoken
```

## Usage

### Stock Forecasts

To generate forecasts for multiple stocks:

```bash
python multi_forecast.py --assets AAPL,MSFT,GOOGL,AMZN,META --type stock --timeframe "next month"
```

### Cryptocurrency Forecasts

To generate forecasts for multiple cryptocurrencies:

```bash
python multi_forecast.py --assets BTC,ETH,SOL,ADA,DOT --type crypto --timeframe "next week"
```

### Advanced Options

You can customize the forecast direction and other parameters:

```bash
python multi_forecast.py --assets AAPL,MSFT,GOOGL --type stock --timeframe "end of year" --direction decrease --num_checks 5 --model gpt-4o --max_workers 3
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--assets` | Comma-separated list of asset symbols | Required |
| `--type` | Type of assets: stock or crypto | Required |
| `--timeframe` | Timeframe for forecast | "next month" |
| `--direction` | Direction of forecast (increase, decrease, change) | "increase" |
| `--num_checks` | Number of consistency checks to run | 3 |
| `--model` | Model to use for forecasting | "o3" |
| `--max_workers` | Number of parallel forecasts to run | 3 |
| `--verbose` | Show verbose output | False |

## Parallel Processing

The tool now supports parallel processing to generate multiple forecasts simultaneously:

- Utilizes Python's `concurrent.futures` ThreadPoolExecutor for parallel execution
- Default of 3 concurrent workers, but can be adjusted with `--max_workers`
- Intelligently distributes workload to maximize efficiency
- Shows performance metrics comparing parallel vs. sequential execution time

Example with parallel processing:

```bash
# Run 5 crypto forecasts with 3 parallel workers
./run_multi_forecast.sh --crypto --assets "BTC,ETH,SOL,DOGE,ADA" --workers 3

# Run 10 stock forecasts with 5 parallel workers for maximum speed
python multi_forecast.py --assets "AAPL,MSFT,GOOGL,AMZN,META,NVDA,INTC,AMD,TSLA,JPM" --type stock --max_workers 5
```

The script will display time saved through parallelization at the end of execution:

```
Completed forecasts for 5 assets in 157.32 seconds.
Average time per asset: 31.46 seconds
Estimated sequential execution time: 283.65 seconds
Time saved through parallelization: 126.33 seconds (44.5%)
```

## Output

The script outputs forecasts sorted by confidence, with the most confident predictions at the top.
For each asset, it shows:

- Current price (if available)
- Forecast result
- Confidence level
- Key factors affecting the prediction
- Execution time

### API Cost Tracking

The tool includes precise API cost estimation using the tiktoken library:

- Accurately tracks input and output tokens used during forecasting
- Calculates costs based on current OpenAI pricing for the selected model
- Displays a detailed cost summary at the end of the forecasting run

#### Supported Models and Pricing

The following models are supported with current pricing (per 1M tokens):

| Model | Input Cost | Output Cost |
|-------|------------|-------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o mini | $0.15 | $0.60 |
| o3 (GPT-4o Omni) | $10.00 | $40.00 |
| GPT-4.1 | $2.00 | $8.00 |
| GPT-4.1 mini | $0.40 | $1.60 |
| GPT-4.1 nano | $0.10 | $0.40 |
| o1 (Claude Opus) | $15.00 | $60.00 |
| o1-mini | $1.10 | $4.40 |

See `utils/token_utils.py` for a complete list of supported models and their pricing.

Example API usage summary:
```
================================================================================
API USAGE SUMMARY
================================================================================
Model: GPT-4o Omni (o3)
Total tokens used: 89,325 tokens
  - Input tokens:  56,428 tokens
  - Output tokens: 32,897 tokens
API calls: 22
Elapsed time: 186.42 seconds

Pricing information:
  - Input price:  $10.00 per 1M tokens
  - Output price: $40.00 per 1M tokens

Estimated cost:
  - Input cost:  $0.5643
  - Output cost: $0.9869
  - Total cost:  $1.5512
================================================================================
```

## Implementation Details

### Token Tracking Utility

The tool uses a specialized token tracking utility found in `utils/token_utils.py` that provides:

- Accurate token counting using the tiktoken library
- Token tracking decorator for monitoring API calls
- Specialized counting for message formatting overhead
- Detailed cost calculation with actual OpenAI pricing

For projects using this code, the token tracking can be used independently:

```python
from utils.token_utils import TokenTracker, count_tokens

# Create a tracker for a specific model
tracker = TokenTracker(model="gpt-4o")

# Add token counts when making API calls
tracker.add_tokens(input_tokens=500, output_tokens=150)

# Get the cost information
cost_info = tracker.get_cost()
print(f"Total cost: ${cost_info['total_cost']:.4f}")

# Or print a complete summary
tracker.print_summary()
``` 