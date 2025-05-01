#!/bin/bash
# Script to run multi-asset forecasting with common configurations
#
# Features:
# - Easy command line options for stock and crypto forecasting
# - Default asset lists for quick testing
# - Timeframe and forecast direction customization
# - API cost tracking and estimation based on token usage
# - Results sorted by confidence level

# Helper function for display
print_header() {
  echo "====================================================="
  echo " $1"
  echo "====================================================="
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Set directory to the location of this script
cd "$(dirname "$0")"

# Options
FORECAST_TYPE=""
ASSETS=""
TIMEFRAME="next month"
DIRECTION="increase"
NUM_CHECKS=3
MODEL="o3"
VERBOSE=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --stocks)
      FORECAST_TYPE="stock"
      shift
      ;;
    --crypto)
      FORECAST_TYPE="crypto"
      shift
      ;;
    --assets)
      ASSETS="$2"
      shift
      shift
      ;;
    --timeframe)
      TIMEFRAME="$2"
      shift
      shift
      ;;
    --direction)
      DIRECTION="$2"
      shift
      shift
      ;;
    --checks)
      NUM_CHECKS="$2"
      shift
      shift
      ;;
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Check required arguments
if [ -z "$FORECAST_TYPE" ]; then
  print_header "ERROR: Missing forecast type"
  echo "Please specify either --stocks or --crypto"
  exit 1
fi

if [ -z "$ASSETS" ]; then
  # Set default assets based on type
  if [ "$FORECAST_TYPE" = "stock" ]; then
    ASSETS="AAPL,MSFT,GOOGL,AMZN,META"
    echo "Using default stock assets: $ASSETS"
  else
    ASSETS="BTC,ETH,SOL,DOGE,ADA"
    echo "Using default crypto assets: $ASSETS"
  fi
fi

# Display configuration
print_header "MULTI-ASSET FORECAST CONFIGURATION"
echo "Type:      $FORECAST_TYPE"
echo "Assets:    $ASSETS"
echo "Timeframe: $TIMEFRAME"
echo "Direction: $DIRECTION"
echo "Checks:    $NUM_CHECKS"
echo "Model:     $MODEL"
echo "Verbose:   ${VERBOSE:-No}"
echo

# Run the forecasting tool
print_header "RUNNING FORECAST"
python multi_forecast.py \
  --assets "$ASSETS" \
  --type "$FORECAST_TYPE" \
  --timeframe "$TIMEFRAME" \
  --direction "$DIRECTION" \
  --num_checks "$NUM_CHECKS" \
  --model "$MODEL" \
  $VERBOSE

# Make the script executable
chmod +x run_multi_forecast.sh 