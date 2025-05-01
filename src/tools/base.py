from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Tool(ABC):
    """Base abstract class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tool."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the tool."""
        pass
        
    @property
    def parameters(self) -> Dict[str, Any]:
        """Return the parameters for the tool."""
        return {}
        
    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """Run the tool with the provided arguments."""
        pass
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool and return a structured result.
        
        Returns:
            A dictionary with 'result' containing the main output
            and optional additional metadata.
        """
        try:
            result = self._run(**kwargs)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 