#!/usr/bin/env python3
"""
Utility functions for analyzing stock-related forecasting questions.
"""

import re
from typing import Dict, Optional, Tuple, List

def is_stock_question(question: str) -> bool:
    """
    Determine if a question is about stocks.
    
    Args:
        question: The forecasting question
        
    Returns:
        True if the question is likely about stocks, False otherwise
    """
    stock_patterns = [
        r'stock\s+price',
        r'share\s+price',
        r'\$[A-Z]+',  # $AAPL format
        r'[A-Z]{1,5}\s+stock',  # AAPL stock format
        r'cross\s+\$?\d+',  # cross $100 format
        r'(above|below|reach|hit)\s+\$?\d+',  # above $100 format
    ]
    
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in stock_patterns)

def extract_ticker_symbol(question: str) -> Optional[str]:
    """
    Extract a ticker symbol from a question.
    
    Args:
        question: The forecasting question
        
    Returns:
        The ticker symbol if found, None otherwise
    """
    # Look for ticker symbols in common formats
    patterns = [
        r'\$([A-Z]{1,5})',  # $AAPL format
        r'([A-Z]{1,5})\s+stock',  # AAPL stock format
        r'([A-Z]{1,5})\s+(shares|price)',  # AAPL shares/price format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Look for common company names and map to tickers
    company_to_ticker = {
        'apple': 'AAPL',
        'amazon': 'AMZN',
        'google': 'GOOGL',
        'microsoft': 'MSFT',
        'tesla': 'TSLA',
        'nvidia': 'NVDA',
        'amd': 'AMD',
        'intel': 'INTC',
        'netflix': 'NFLX',
        'meta': 'META',
        'facebook': 'META'
    }
    
    for company, ticker in company_to_ticker.items():
        if re.search(r'\b' + company + r'\b', question, re.IGNORECASE):
            return ticker
    
    # Last resort: look for any capital letter sequence that might be a ticker
    match = re.search(r'\b([A-Z]{1,5})\b', question)
    if match:
        return match.group(1)
    
    return None

def extract_price_threshold(question: str) -> Optional[float]:
    """
    Extract a price threshold from a question.
    
    Args:
        question: The forecasting question
        
    Returns:
        The price threshold if found, None otherwise
    """
    # Look for price thresholds in various formats
    patterns = [
        r'cross\s+\$?(\d+(?:\.\d+)?)',  # cross $100 format
        r'(above|below|reach|hit)\s+\$?(\d+(?:\.\d+)?)',  # above $100 format
        r'\$(\d+(?:\.\d+)?)',  # $100 format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            # If the pattern has a group for the direction, the price is in group 2
            if 'above|below|reach|hit' in pattern:
                return float(match.group(2))
            else:
                return float(match.group(1))
    
    return None

def analyze_stock_question(question: str) -> Dict:
    """
    Analyze a stock-related question and extract key information.
    
    Args:
        question: The forecasting question
        
    Returns:
        Dictionary containing extracted information
    """
    if not is_stock_question(question):
        return {"is_stock_question": False}
    
    result = {
        "is_stock_question": True,
        "ticker": extract_ticker_symbol(question),
        "price_threshold": extract_price_threshold(question),
    }
    
    # Determine direction (cross above vs cross below)
    direction_match = re.search(r'(cross|go|fall|drop|rise|climb|jump|increase|decrease)\s+(above|below|under|over)', question, re.IGNORECASE)
    if direction_match:
        direction = direction_match.group(2).lower()
        if direction in ['above', 'over']:
            result["direction"] = "above"
        elif direction in ['below', 'under']:
            result["direction"] = "below"
    else:
        # If no explicit direction, try to infer from other words
        if re.search(r'(rise|climb|jump|increase|rally|gain|up)', question, re.IGNORECASE):
            result["direction"] = "above"
        elif re.search(r'(fall|drop|decrease|decline|down)', question, re.IGNORECASE):
            result["direction"] = "below"
        else:
            # Default interpretation for "cross" without direction
            result["direction"] = "cross_either"
    
    # Extract time frame
    timeframe_match = re.search(r'(this|next|coming|following)\s+(day|week|month|year|quarter)', question, re.IGNORECASE)
    if timeframe_match:
        result["timeframe"] = f"{timeframe_match.group(1)} {timeframe_match.group(2)}"
    
    return result
    
if __name__ == "__main__":
    # Test examples
    test_questions = [
        "Will AMD stock cross 110 dollars next month?",
        "Is NVDA going to hit $200 this year?",
        "What's the probability that Apple shares go below $150 in the next quarter?",
        "Will Amazon stock price increase by 10% next week?",
        "What are the chances TSLA crosses $400 by the end of the year?"
    ]
    
    for q in test_questions:
        analysis = analyze_stock_question(q)
        print(f"Question: {q}")
        print(f"Analysis: {analysis}")
        print() 