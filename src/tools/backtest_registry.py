from typing import Optional
from .base import Tool
from .backtest_web_search import BacktestWebSearchTool

# Define a local copy of ToolRegistry rather than importing it
class BacktestToolRegistry:
    """Registry for time-bounded tools used in backtesting."""
    
    def __init__(self, cutoff_date: str, api_key: Optional[str] = None, cx: Optional[str] = None):
        """
        Initialize the backtest tool registry.
        
        Args:
            cutoff_date: Date in YYYY-MM-DD format representing the knowledge cutoff
            api_key: Google API key. If None, will look for GOOGLE_API_KEY env variable
            cx: Google Custom Search Engine ID. If None, will look for GOOGLE_CX env variable
        """
        self._tools = {}
        self.cutoff_date = cutoff_date
        
        # Set up time-bounded tools
        try:
            # Initialize and register the backtest web search tool
            web_search = BacktestWebSearchTool(
                cutoff_date=cutoff_date,
                api_key=api_key,
                cx=cx
            )
            self.register(web_search)
            print(f"[Backtest] Registered time-bounded web search tool with cutoff date: {cutoff_date}")
        except ValueError as e:
            print(f"[Backtest] Warning: Could not register web search tool: {e}")
            
    def register(self, tool: Tool) -> None:
        """Register a tool instance with the registry."""
        self._tools[tool.name] = tool
        
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def list_tools(self) -> list:
        """List all registered tools."""
        return list(self._tools.values())
        
    def get_tool_descriptions(self) -> list:
        """Get descriptions of all tools for LLM tool calling."""
        descriptions = []
        for tool in self._tools.values():
            descriptions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": list(tool.parameters.keys())
                    }
                }
            })
        return descriptions 