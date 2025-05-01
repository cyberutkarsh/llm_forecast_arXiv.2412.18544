import os
import json
import openai
from typing import Dict, Any, List, Optional

class QueryClassifier:
    """
    Uses LLM to classify user queries and determine the appropriate tool to handle them.
    This eliminates the need for hardcoded pattern matching and makes the system more flexible.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the query classifier.
        
        Args:
            model: The OpenAI model to use for classification
        """
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def classify_query(self, query: str, available_tools: List[str]) -> Dict[str, Any]:
        """
        Classify a query to determine the appropriate tool and extract relevant parameters.
        
        Args:
            query: The user's query
            available_tools: List of available tool names
            
        Returns:
            Dictionary with classification information
        """
        tools_str = ", ".join(available_tools)
        
        # Create a prompt that describes each tool's purpose
        tool_descriptions = self._get_tool_descriptions(available_tools)
        
        prompt = f"""
        Analyze this query and determine which tool would be most appropriate to handle it:
        
        Query: "{query}"
        
        Available tools:
        {tool_descriptions}
        
        For the selected tool, extract any relevant parameters from the query.
        
        Return your analysis as a JSON object with these fields:
        {{
            "selected_tool": the name of the most appropriate tool from the available list,
            "confidence": a number between 0 and 1 indicating how confident you are in this selection,
            "reason": a brief explanation of why this tool is appropriate,
            "parameters": {{
                // Parameters relevant to the selected tool
                // For web_search: "query" (the search query to use)
                // For other tools: tool-specific parameters
            }}
        }}
        
        Analyze the query carefully and select the most appropriate tool based on the user's intent.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            try:
                classification = json.loads(content)
                
                # Validate that the selected tool is in the available tools list
                if classification.get("selected_tool") not in available_tools:
                    print(f"[Classifier] Warning: Selected tool {classification.get('selected_tool')} not in available tools list")
                    classification["selected_tool"] = "web_search"  # Fallback to web search
                
                return classification
            except json.JSONDecodeError:
                print(f"[Classifier] Error parsing LLM classification")
                # Fallback classification
                return {
                    "selected_tool": "web_search",
                    "confidence": 0.5,
                    "reason": "Fallback due to parsing error",
                    "parameters": {"query": query}
                }
        except Exception as e:
            print(f"[Classifier] Error using LLM for query classification: {e}")
            # Fallback classification
            return {
                "selected_tool": "web_search",
                "confidence": 0.5,
                "reason": f"Fallback due to error: {str(e)}",
                "parameters": {"query": query}
            }
    
    def _get_tool_descriptions(self, available_tools: List[str]) -> str:
        """
        Get descriptions for the available tools.
        
        Args:
            available_tools: List of available tool names
            
        Returns:
            String with tool descriptions
        """
        tool_descriptions = {
            "web_search": "Searches the web for information using Google Custom Search. Use for general information retrieval, current events, facts, etc.",
            "stock_data": "Retrieves current stock data for a specific ticker symbol. Use for current stock prices, market caps, P/E ratios, etc.",
            "crypto_data": "Retrieves current cryptocurrency data. Use for current crypto prices, market caps, 24h volumes, etc.",
            "weather_data": "Retrieves current weather data for a specific location. Use for temperature, conditions, forecasts, etc.",
            "calculator": "Performs mathematical calculations. Use for complex math operations, conversions, etc.",
            "date_calculator": "Performs date-related calculations. Use for finding days between dates, adding/subtracting periods, etc."
        }
        
        descriptions = []
        for tool in available_tools:
            if tool in tool_descriptions:
                descriptions.append(f"- {tool}: {tool_descriptions[tool]}")
            else:
                descriptions.append(f"- {tool}: Generic tool")
        
        return "\n".join(descriptions) 