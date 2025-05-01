#!/usr/bin/env python3
"""
Utility module for token counting and cost calculation.
"""

import os
import functools
import time
from typing import Dict, Any, Callable, List, Optional
import tiktoken
from functools import lru_cache

# Current OpenAI API pricing as of May 2024
# Source: https://openai.com/api/pricing/
MODEL_PRICING = {
    # GPT-4o family
    "gpt-4o": {"input": 2.50, "output": 10.00, "name": "GPT-4o"},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00, "name": "GPT-4o"},
    "o3": {"input": 10.00, "output": 40.00, "name": "GPT-4o Omni (o3)"},
    "o3-2025-04-16": {"input": 10.00, "output": 40.00, "name": "GPT-4o Omni (o3)"},
    
    # GPT-4o mini family
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "name": "GPT-4o mini"},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60, "name": "GPT-4o mini"},
    
    # GPT-4.1 family
    "gpt-4.1": {"input": 2.00, "output": 8.00, "name": "GPT-4.1"},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00, "name": "GPT-4.1"},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "name": "GPT-4.1 mini"},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60, "name": "GPT-4.1 mini"},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "name": "GPT-4.1 nano"},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40, "name": "GPT-4.1 nano"},
    
    # GPT-4.5 preview
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00, "name": "GPT-4.5 preview"},
    "gpt-4.5-preview-2025-02-27": {"input": 75.00, "output": 150.00, "name": "GPT-4.5 preview"},
    
    # Claude family (o1, o1-mini)
    "o1": {"input": 15.00, "output": 60.00, "name": "Claude Opus (o1)"},
    "o1-2024-12-17": {"input": 15.00, "output": 60.00, "name": "Claude Opus (o1)"},
    "o1-pro": {"input": 150.00, "output": 600.00, "name": "Claude Opus Pro"},
    "o1-pro-2025-03-19": {"input": 150.00, "output": 600.00, "name": "Claude Opus Pro"},
    "o1-mini": {"input": 1.10, "output": 4.40, "name": "Claude Opus Mini"},
    "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40, "name": "Claude Opus Mini"},
    
    # GPT-4o specialized variants
    "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60, "name": "GPT-4o mini search preview"},
    "gpt-4o-search-preview": {"input": 2.50, "output": 10.00, "name": "GPT-4o search preview"},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00, "name": "GPT-4o audio preview"},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60, "name": "GPT-4o mini audio preview"},
    "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00, "name": "GPT-4o realtime preview"},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40, "name": "GPT-4o mini realtime preview"},
    
    # Other o-series models
    "o4-mini": {"input": 1.10, "output": 4.40, "name": "o4 mini"},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40, "name": "o4 mini"},
    "o3-mini": {"input": 1.10, "output": 4.40, "name": "o3 mini"},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40, "name": "o3 mini"},
    
    # Fine-tuned models - approximate pricing based on base models
    "ft:gpt-4o": {"input": 2.50, "output": 10.00, "name": "Fine-tuned GPT-4o"},
    
    # Legacy models - kept for backward compatibility
    "gpt-4": {"input": 30.00, "output": 60.00, "name": "GPT-4 (Legacy)"},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "name": "GPT-4 Turbo (Legacy)"},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "name": "GPT-3.5 Turbo"},
    
    # Computer & vision models
    "computer-use-preview": {"input": 3.00, "output": 12.00, "name": "Computer Use Preview"},
    "gpt-image-1": {"input": 5.00, "output": 0.00, "name": "GPT Image-1"}
}

@lru_cache(maxsize=10)
def get_token_encoder(model_name: str):
    """
    Get the appropriate tiktoken encoder for a given model.
    Uses LRU cache to avoid repeatedly loading encoders.
    """
    try:
        # For GPT-4 models
        if model_name.startswith(("gpt-4", "o3")):
            return tiktoken.encoding_for_model("gpt-4")
        # For GPT-3.5 models
        elif model_name.startswith("gpt-3.5"):
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        # Default fallback
        else:
            return tiktoken.encoding_for_model("cl100k_base")  # The most common encoding
    except Exception:
        # Fallback to cl100k_base if specific encoding not found
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in the given text for the specified model.
    
    Args:
        text: The text to count tokens for
        model: The model to use for token counting
        
    Returns:
        Number of tokens
    """
    encoder = get_token_encoder(model)
    tokens = encoder.encode(text)
    return len(tokens)

def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o") -> Dict[str, int]:
    """
    Count tokens in a list of messages for the specified model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name to use for token counting
        
    Returns:
        Dictionary with token counts
    """
    encoder = get_token_encoder(model)
    
    # Initialize token count for each message
    total_tokens = 0
    
    # Count tokens per message
    for message in messages:
        # Count tokens in the message content
        content = message.get("content", "")
        if isinstance(content, str):
            total_tokens += len(encoder.encode(content))
        elif isinstance(content, list):  # For multi-modal content
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total_tokens += len(encoder.encode(item["text"]))
        
        # Add tokens for message role
        total_tokens += 4  # Each message uses ~4 tokens for role formatting
    
    # Add tokens for the system message formatting
    total_tokens += 3  # Every API call has ~3 tokens of system formatting
    
    return total_tokens

def calculate_cost(tokens: Dict[str, int], model: str = "gpt-4o") -> Dict[str, float]:
    """
    Calculate cost for token usage with the given model.
    
    Args:
        tokens: Dictionary with 'input' and 'output' token counts
        model: The model name to calculate pricing for
        
    Returns:
        Dictionary with cost breakdown
    """
    # Get pricing for the model
    model_lower = model.lower()
    pricing = MODEL_PRICING.get(model_lower, MODEL_PRICING.get("gpt-4o"))  # Default to gpt-4o if not found
    
    # Calculate costs per token category
    input_cost = (tokens.get("input", 0) / 1_000_000) * pricing["input"]
    output_cost = (tokens.get("output", 0) / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model_name": pricing["name"],
        "input_price_per_1m": pricing["input"],
        "output_price_per_1m": pricing["output"]
    }

class TokenTracker:
    """
    Utility class for tracking token usage and costs across multiple API calls.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize a token tracker.
        
        Args:
            model: The model to track tokens for
        """
        self.model = model
        self.total_tokens = {"input": 0, "output": 0}
        self.api_calls = 0
        self.start_time = time.time()
        
    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """
        Add token counts to the tracker.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.total_tokens["input"] += input_tokens
        self.total_tokens["output"] += output_tokens
        self.api_calls += 1
        
    def get_token_counts(self) -> Dict[str, int]:
        """
        Get the current token counts.
        
        Returns:
            Dictionary with token counts
        """
        return {
            "input": self.total_tokens["input"],
            "output": self.total_tokens["output"],
            "total": self.total_tokens["input"] + self.total_tokens["output"],
            "api_calls": self.api_calls
        }
        
    def get_cost(self) -> Dict[str, Any]:
        """
        Get the estimated cost of the token usage.
        
        Returns:
            Dictionary with cost breakdown
        """
        cost_info = calculate_cost(self.total_tokens, self.model)
        elapsed_time = time.time() - self.start_time
        
        return {
            **cost_info,
            "api_calls": self.api_calls,
            "elapsed_time": elapsed_time
        }
        
    def reset(self) -> None:
        """Reset the token counters."""
        self.total_tokens = {"input": 0, "output": 0}
        self.api_calls = 0
        self.start_time = time.time()
        
    def print_summary(self) -> None:
        """Print a summary of token usage and cost."""
        cost_info = self.get_cost()
        total_tokens = self.total_tokens["input"] + self.total_tokens["output"]
        
        print("\n" + "="*80)
        print(f"API USAGE SUMMARY")
        print("="*80)
        print(f"Model: {cost_info['model_name']}")
        print(f"Total tokens used: {total_tokens:,} tokens")
        print(f"  - Input tokens:  {self.total_tokens['input']:,} tokens")
        print(f"  - Output tokens: {self.total_tokens['output']:,} tokens")
        print(f"API calls: {self.api_calls}")
        print(f"Elapsed time: {cost_info['elapsed_time']:.2f} seconds")
        print("\nPricing information:")
        print(f"  - Input price:  ${cost_info['input_price_per_1m']:.2f} per 1M tokens")
        print(f"  - Output price: ${cost_info['output_price_per_1m']:.2f} per 1M tokens")
        print("\nEstimated cost:")
        print(f"  - Input cost:  ${cost_info['input_cost']:.4f}")
        print(f"  - Output cost: ${cost_info['output_cost']:.4f}")
        print(f"  - Total cost:  ${cost_info['total_cost']:.4f}")
        print("="*80)

def track_tokens(model: str = "gpt-4o"):
    """
    Decorator for tracking tokens used in an LLM call function.
    
    Args:
        model: Model being used
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the token tracker instance, likely on the first arg (self)
            instance = args[0]
            if not hasattr(instance, "token_tracker"):
                instance.token_tracker = TokenTracker(model)
            
            # Call the original function
            response = func(*args, **kwargs)
            
            # Extract token usage from the response
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                instance.token_tracker.add_tokens(
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0)
                )
            
            return response
        return wrapper
    return decorator 