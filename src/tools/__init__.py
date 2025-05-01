from typing import Dict, List, Type, Optional
from .base import Tool
from .web_search import WebSearchTool
from .backtest_web_search import BacktestWebSearchTool
from .backtest_registry import BacktestToolRegistry

class ToolRegistry:
    """Registry for managing and accessing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool) -> None:
        """Register a tool instance with the registry."""
        self._tools[tool.name] = tool
        
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered by name."""
        return name in self._tools
        
    def deregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
        
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
        
    def get_tool_descriptions(self) -> List[Dict[str, any]]:
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

# Create default registry with standard tools
default_registry = ToolRegistry()

# Export classes
__all__ = ["Tool", "WebSearchTool", "BacktestWebSearchTool", "ToolRegistry", "BacktestToolRegistry", "default_registry"] 