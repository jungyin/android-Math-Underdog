import asyncio
import json
from typing import Dict, List, Any, Optional
import websockets
from ..utils.base_protocol import MCPRequest, MCPResponse, ToolDescription
from ..utils.cache_response import ResponseListener

class MCPServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.tools: Dict[str, ToolDescription] = {}
        self.client_listeners: Dict[websockets.WebSocketServerProtocol, ResponseListener] = {}


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
        print("method",method)
        print("self.tools",self.tools)
        for client_id, client in self.clients.items():
            tool_key = f"{client_id}:{method}"
            print("check_tool_key",tool_key)
            if tool_key in self.tools:
                try:
                    # 创建工具调用请求
                    request = MCPRequest(
                        method=method,
                        params=params,
                        id=f"server_call_{method}"
                    )
                    # 发送请求给客户端
                    future = asyncio.Future()

                    try:
                        listener =  self.client_listeners[client]
                        # 尝试注册 Future，如果 ID 冲突会抛异常
                        # 这一步必须在 send 之前完成
                        listener.register_future(request.id, future) 
                        print("request",request)
                        await client.send(json.dumps(vars(request)))
                        
                        # 发送后立即等待 Future 被设置结果
                        response_data = await asyncio.wait_for(future, timeout=10)
                        
                        listener.unregister_future(request.id)
                        print("处理成功",response_data)
                        return response_data
                    except asyncio.TimeoutError:
                        print(f"[服务器] 等待请求 {request.id} 的响应超时。")
                        # 超时时，确保从监听器中移除 Future
                        listener.unregister_future(request.id) 
                    except ValueError as e: # 捕获 register_future 可能抛出的 ID 冲突异常
                        print(f"[服务器] 注册 Future 失败: {e}")
                        listener.unregister_future(request.id)
                    except Exception as e:
                        print(f"[服务器] 等待请求 {request.id} 响应时发生错误: {e}")
                        # 确保异常时也清理
                        import traceback
                        traceback.print_exc() # 打印完整的异常堆栈
                        listener.unregister_future(request.id)
                    finally:
                        # 正常完成或超时后，future 会在 register_for_response 内部或 TimeoutError 捕获中移除
                        # 这里不需要重复移除，但如果 Future 因其他原因未被清除，则需要考虑
                        pass # 由 listener 内部管理移除
                
                    return None
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
            print("一个客户端接入",websocket.remote_address)
            self.client_listeners[websocket] = ResponseListener()
            async for message in websocket:
                
                print("一个请求接入",websocket.remote_address)
                print("message",message)
                data = json.loads(message)
                run = True
                # 通过添加这个判断，来防止mcp_clinet发送回的消息再进入这个message队列
                if not (self.client_listeners[websocket] is None)  :
                    # 这里判断一下，如果说别的地方在监听，就在这里拦截返回一下
                    if(self.client_listeners[websocket].deliver_message(data)):
                        run = False 
                        
                if run:
                    request = MCPRequest(**data)
                    print("request",request)
                    print("请求接口",request.method)
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
                            response = MCPResponse(result=tool_result['result'], id=request.id)
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
            print("进来了？")
            # 处理连接关闭
            for client_id, client in self.clients.items():
                if client == websocket:
                    if websocket in self.client_listeners:
                        self.client_listeners[websocket].cleanup()
                        del self.client_listeners[websocket]
                    await self.unregister_client(client_id)
                    break

    async def start(self):
        async with websockets.serve(self.handle_client_message, self.host, self.port):
            print(f"MCP Server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # 运行直到被中断

if __name__ == "__main__":
    server = MCPServer()
    asyncio.run(server.start()) 