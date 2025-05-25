import asyncio
import json
from typing import Dict, List, Any, Optional
import websockets
from ..utils.base_protocol import MCPRequest, MCPResponse, ToolDescription

class MCPServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.tools: Dict[str, ToolDescription] = {}
        self.tool_implementations: Dict[str, Dict[str, Any]] = {}

    async def register_client(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        self.clients[client_id] = websocket
        print(f"Client {client_id} registered")

    async def unregister_client(self, client_id: str):
        if client_id in self.clients:
            del self.clients[client_id]
            print(f"Client {client_id} unregistered")

    def register_tool(self, client_id: str, tool: ToolDescription):
        """注册工具到服务器"""
        tool_key = f"{client_id}:{tool.name}"
        self.tools[tool_key] = tool
        print(f"Tool {tool.name} registered by client {client_id}")

    async def call_tool(self, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """调用工具并获取结果"""
        # 查找实现了该工具的客户端
        for client_id, client in self.clients.items():
            tool_key = f"{client_id}:{method}"
            if tool_key in self.tools:
                try:
                    # 创建工具调用请求
                    request = MCPRequest(
                        method=method,
                        params=params,
                        id=f"server_call_{method}"
                    )
                    # 发送请求给客户端
                    await client.send(json.dumps(vars(request)))
                    # 等待响应
                    response = await client.recv()
                    response_data = json.loads(response)
                    return MCPResponse(**response_data)
                except Exception as e:
                    print(f"Tool call error: {e}")
                    return None
        return None

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取所有可用工具的描述"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "required_params": tool.required_params
            }
            for tool in self.tools.values()
        ]

    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol):
        try:
            async for message in websocket:
                data = json.loads(message)
                request = MCPRequest(**data)
                
                if request.method == "register":
                    await self.register_client(websocket, request.params["client_id"])
                    response = MCPResponse(result="registered", id=request.id)
                elif request.method == "register_tool":
                    # 处理工具注册
                    tool_desc = ToolDescription(**request.params["tool"])
                    self.register_tool(request.params["client_id"], tool_desc)
                    response = MCPResponse(result="tool_registered", id=request.id)
                elif request.method == "list_tools":
                    # 返回所有可用工具列表
                    tools_list = self.get_available_tools()
                    response = MCPResponse(result=tools_list, id=request.id)
                elif request.method == "call_tool":
                    # 处理工具调用
                    tool_result = await self.call_tool(
                        request.params["tool_name"],
                        request.params.get("tool_params", {})
                    )
                    if tool_result:
                        response = MCPResponse(result=tool_result.result, id=request.id)
                    else:
                        response = MCPResponse(
                            result=None,
                            error="Tool call failed or tool not found",
                            id=request.id
                        )
                else:
                    # 转发工具调用给对应的客户端
                    tool_response = await self.call_tool(request.method, request.params)
                    if tool_response:
                        response = tool_response
                    else:
                        response = MCPResponse(
                            result=None,
                            error=f"Unknown method or tool not available: {request.method}",
                            id=request.id
                        )
                
                await websocket.send(json.dumps(vars(response)))
        except websockets.exceptions.ConnectionClosed:
            # 处理连接关闭
            for client_id, client in self.clients.items():
                if client == websocket:
                    await self.unregister_client(client_id)
                    break

    async def start(self):
        async with websockets.serve(self.handle_client_message, self.host, self.port):
            print(f"MCP Server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # 运行直到被中断

if __name__ == "__main__":
    server = MCPServer()
    asyncio.run(server.start()) 