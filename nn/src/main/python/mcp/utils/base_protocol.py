from dataclasses import dataclass
from typing import Any, Dict, List, Optional
@dataclass
class MCPRequest:
    """MCP请求结构"""
    method: str
    params: Dict[str, Any]
    id: str

@dataclass
class MCPResponse:
    """MCP响应结构"""
    result: Any
    error: Optional[str] = None
    id: Optional[str] = None

@dataclass
class ToolDescription:
    """工具描述结构"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] 


