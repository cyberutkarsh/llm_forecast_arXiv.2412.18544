# LLM Forecasting with Consistency Checks

This project implements a terminal-based forecasting system using large language models with consistency checks as described in ["Consistency Checks for Language Model Forecasters"](https://arxiv.org/abs/2402.05458).

## Setup

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key (required):
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### External API Setup (Optional)

The system can utilize several external APIs to enhance forecasting with real-time data:

#### 1. Web Search Tool

To enable real-time web information retrieval:

```
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_google_custom_search_engine_id_here
```

- [Google Custom Search API](https://developers.google.com/custom-search/v1/introduction)
- [Programmable Search Engine](https://programmablesearchengine.google.com/)

#### 2. Stock Data Tool

To enable real-time stock data:

```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

- [Alpha Vantage API](https://www.alphavantage.co/support/#api-key) (free tier available)

#### 3. Cryptocurrency Data Tool

This tool uses the free CoinGecko API and does not require an API key.

## Usage

Run the forecasting tool with:

```bash
python src/main.py --question "Will global temperatures rise by more than 2°C by 2030?"
```

Options:
- `--question`: The forecasting question to ask
- `--model`: LLM model to use (default: gpt-4o/o3)
- `--num_checks`: Number of consistency checks to perform (default: 3)
- `--verbose`: Show detailed chain of thought (default: False)
- `--disable_web_search`: Disable web search functionality (default: False)
- `--disable_stock_data`: Disable stock data functionality (default: False)
- `--disable_crypto_data`: Disable cryptocurrency data functionality (default: False)
- `--disable_cache`: Disable response caching (default: False)

### Examples

For questions about stocks:

```bash
python src/main.py --question "Will NVDA stock reach $1200 next month?" --verbose
```

For questions about cryptocurrencies:

```bash
python src/main.py --question "Will Bitcoin exceed $100,000 by the end of this year?" --verbose
```

For general forecasting questions with web search:

```bash
python src/main.py --question "Is it likely that atmospheric CO2 will exceed 450ppm by 2030?" --verbose
```

### Multi-Asset Forecasting

For forecasting multiple assets at once, use the multi-asset forecasting tool:

```bash
# Forecast multiple stocks
python test/examples/multi_forecast.py --assets AAPL,MSFT,GOOGL,AMZN,META --type stock

# Forecast multiple cryptocurrencies
python test/examples/multi_forecast.py --assets BTC,ETH,SOL,DOGE,ADA --type crypto
```

Or use the provided shell script for convenience:

```bash
# For stocks
./test/examples/run_multi_forecast.sh --stocks --assets "AAPL,MSFT,NVDA,AMD,INTC"

# For cryptocurrencies
./test/examples/run_multi_forecast.sh --crypto --assets "BTC,ETH,SOL,DOT,ADA"
```

See `test/examples/README.md` for more details on multi-asset forecasting options.

## Implementation Details

The system implements several types of consistency checks:
1. **Different decompositions**: Breaking down the question in multiple ways
2. **Different chain-of-thought paths**: Generating multiple reasoning paths
3. **Different framings**: Rephrasing the question in multiple ways

### Tool Support

The system includes an expandable tool framework:

- **Web Search**: Uses Google's Custom Search API to retrieve real-time information
- **Stock Data**: Uses Alpha Vantage API for current stock prices and market data
- **Crypto Data**: Uses CoinGecko API for current cryptocurrency prices and market data
- **LLM-Based Query Classification**: Routes questions to the appropriate tool

All data extraction is performed using LLMs without hardcoded patterns, making the system flexible across various domains.

## Features

- Terminal-based forecasting using OpenAI language models
- Structured forecasting output with reasoning, evidence, and predictions
- Consistency checks for probabilities
- Real-time financial and general information access
- Support for multiple forecast variations
- Statistical analysis of consistency across variations

## How It Works

1. **Question Analysis**: The system analyzes the forecasting question using LLMs to identify:
   - The domain (financial, weather, politics, etc.)
   - Key entities (stocks, cryptocurrencies, events)
   - Specific thresholds and timeframes

2. **Tool Selection**: Based on the analysis, appropriate tools are selected:
   - Stock data for stock market questions
   - Crypto data for cryptocurrency questions
   - Web search for general knowledge questions

3. **Data Extraction**: The selected tools retrieve current information:
   - Stock prices and market data from Alpha Vantage
   - Cryptocurrency prices from CoinGecko
   - Web search results via Google Custom Search

4. **Forecasting**: The system generates a forecast using:
   - Retrieved real-time data
   - LLM reasoning with structured prompts
   - Multiple consistency checks to evaluate reliability

5. **Result Analysis**: The system analyzes consistency between different forecasting methods to provide:
   - A final forecast with confidence level
   - Key factors influencing the forecast
   - Consistency score based on agreement between methods

## Consistency Checks

The system implements several key consistency checks:

1. **Temporal Decomposition**: Breaking the forecast down into a sequence of events over time
2. **Spatial Decomposition**: Breaking the forecast down into different components or factors
3. **Chain of Thought**: Using alternative reasoning paths to approach the same question
4. **Alternative Framing**: Rephrasing the question to see if it leads to different perspectives

When running multiple consistency checks, the system calculates a consistency score based on the variance between probability estimates, with higher scores indicating greater agreement.

## Example Output

```
==================================================
Question: Will META cross 600 by the end of this month?
==================================================

Question Analysis:
  Type: financial
  Entity: META
  Identifier: META
  Threshold: 600
  Threshold Unit: dollars
  Direction: above
  Timeframe: by the end of this month
  Event Description: META stock price crossing 600 dollars
==================================================
Final forecast: It is unlikely that META will exceed $600 by the end of this month.
Confidence: High
Execution time: 0.00 seconds

Key factors affecting this forecast:
['Current META price is $312.65', 'Would require a 92% increase in the remaining time', 'Historical volatility for META stock (~2.5% daily)', 'No scheduled catalysts expected', 'Market circuit-breaker rules would prevent such an extreme move']

Consistency Analysis:
- Mean probability: 0.05
- Standard deviation: 0.01
- Range (max - min): 0.03
- Consistency score: 0.99
```

## License

MIT 