# Extending the Forecasting System with Custom Tools

This guide explains how to create and integrate custom tools into the LLM forecasting system.

## Tool System Architecture

The tool system is designed to be modular and extensible. It consists of:

1. **Base Tool Interface**: The abstract `Tool` class that defines the interface for all tools
2. **Tool Registry**: A central registry that manages tool registration and retrieval
3. **Tool Implementation**: Individual tool classes that inherit from the base Tool class

## Creating a Custom Tool

To create a custom tool, follow these steps:

### 1. Create a new tool class

```python
from tools.base import Tool

class MyCustomTool(Tool):
    """Description of what your tool does."""
    
    @property
    def name(self) -> str:
        return "my_custom_tool"  # This name will be used when calling the tool
        
    @property
    def description(self) -> str:
        return "Detailed description of what this tool does and when to use it."
    
    @property
    def parameters(self) -> dict:
        return {
            "param1": {
                "type": "string",
                "description": "Description of the first parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Description of the second parameter"
            }
        }
    
    def _run(self, param1: str, param2: int) -> dict:
        """
        Implement the actual tool logic here.
        
        Args:
            param1: First parameter
            param2: Second parameter
            
        Returns:
            Dictionary with the tool's output
        """
        # Your implementation here
        result = do_something(param1, param2)
        
        return {
            "output": result,
            "metadata": "Additional information"
        }
```

### 2. Register your tool

You can register your tool in different ways:

#### Option 1: Register with the default registry

```python
from tools import default_registry
from my_module import MyCustomTool

# Register the tool with the default registry
default_registry.register(MyCustomTool())
```

#### Option 2: Create and use a custom registry

```python
from tools import ToolRegistry
from my_module import MyCustomTool, AnotherTool

# Create a custom registry
my_registry = ToolRegistry()

# Register tools with your custom registry
my_registry.register(MyCustomTool())
my_registry.register(AnotherTool())

# Use your custom registry with the forecasting system
from forecaster import ForecastingSystem
forecaster = ForecastingSystem(tools_registry=my_registry)
```

### 3. Update the system prompt (optional)

For better results, you might want to update the system prompt to instruct the LLM about your custom tool:

```python
def _get_system_prompt(self) -> str:
    """Get the system prompt for the LLM."""
    system_prompt = """
    You are an AI forecasting assistant with access to tools to help you make accurate predictions.
    
    Your task is to provide probability estimates for questions about future events.
    
    You have access to the following tools:
    - web_search: Search the web for real-time information
    - my_custom_tool: [Description of what your tool does]
    
    When making your final forecast:
    1. Provide step-by-step reasoning
    2. Synthesize information from your knowledge and tool research
    3. Express uncertainty appropriately
    4. Return a numeric probability between 0 and 1
    """
    return system_prompt
```

## Example Tools

### API-Based Tool

```python
import requests
from tools.base import Tool

class WeatherForecastTool(Tool):
    """Tool for retrieving weather forecasts."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1"
    
    @property
    def name(self) -> str:
        return "weather_forecast"
        
    @property
    def description(self) -> str:
        return "Get weather forecasts for a specific location and date."
    
    @property
    def parameters(self) -> dict:
        return {
            "location": {
                "type": "string",
                "description": "City name or ZIP code"
            },
            "days": {
                "type": "integer",
                "description": "Number of days to forecast (1-10)"
            }
        }
    
    def _run(self, location: str, days: int = 3) -> dict:
        """Get weather forecast for the specified location."""
        params = {
            "key": self.api_key,
            "q": location,
            "days": min(days, 10),
            "aqi": "no"
        }
        
        response = requests.get(f"{self.base_url}/forecast.json", params=params)
        
        if response.status_code != 200:
            return {
                "error": f"API error: {response.status_code}",
                "message": response.text
            }
            
        data = response.json()
        
        # Extract relevant information
        forecast = data["forecast"]["forecastday"]
        result = []
        
        for day in forecast:
            result.append({
                "date": day["date"],
                "max_temp_c": day["day"]["maxtemp_c"],
                "min_temp_c": day["day"]["mintemp_c"],
                "condition": day["day"]["condition"]["text"],
                "chance_of_rain": day["day"]["daily_chance_of_rain"]
            })
            
        return {
            "location": data["location"]["name"],
            "country": data["location"]["country"],
            "forecast": result
        }
```

### Computational Tool

```python
import numpy as np
from tools.base import Tool

class StatisticalAnalysisTool(Tool):
    """Tool for performing statistical analysis on numerical data."""
    
    @property
    def name(self) -> str:
        return "statistical_analysis"
        
    @property
    def description(self) -> str:
        return "Perform statistical analysis on a list of numbers."
    
    @property
    def parameters(self) -> dict:
        return {
            "data": {
                "type": "array",
                "description": "List of numerical values to analyze"
            }
        }
    
    def _run(self, data: list) -> dict:
        """Perform statistical analysis on the provided data."""
        if not data:
            return {"error": "Empty data provided"}
            
        # Convert all values to float
        try:
            numerical_data = [float(x) for x in data]
        except (ValueError, TypeError):
            return {"error": "Data contains non-numerical values"}
            
        # Calculate statistics
        return {
            "count": len(numerical_data),
            "mean": float(np.mean(numerical_data)),
            "median": float(np.median(numerical_data)),
            "std_dev": float(np.std(numerical_data)),
            "min": float(np.min(numerical_data)),
            "max": float(np.max(numerical_data)),
            "quartiles": [
                float(np.percentile(numerical_data, 25)),
                float(np.percentile(numerical_data, 50)),
                float(np.percentile(numerical_data, 75))
            ]
        }
```

## Best Practices

1. **Error Handling**: Tools should handle errors gracefully and return informative error messages
2. **Clear Documentation**: Provide clear descriptions of what your tool does and its parameters
3. **Parameter Validation**: Validate input parameters before processing
4. **Return Structured Data**: Return data in a consistent, structured format
5. **API Keys**: Don't hardcode API keys - use environment variables or pass them at initialization
6. **Testing**: Write tests for your tools to ensure they work correctly 