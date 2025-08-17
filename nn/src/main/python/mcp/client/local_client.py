import asyncio
import json
import uuid
import websockets
from ..utils.base_protocol import MCPRequest, MCPResponse, ToolDescription
import requests
from typing import Dict
from ..utils import config

class LocalClient:
    def __init__(self, server_url: str = config.base_url, amap_key: str = "c49485098e88a45c418addbbfe4de800"):
        self.server_url = server_url
        self.client_id = f"local_client_{uuid.uuid4().hex[:8]}"
        self.amap_key = amap_key
        self.websocket = None
        self.tools = self.create_tools()

    def create_tools(self) -> Dict[str, ToolDescription]:
        """创建工具描述"""
        location_tool = ToolDescription(
            name="get_location",
            description="获取当前用户的位置信息，如所在城市，所在国家，所在省份。如果用户未指定地点，将默认使用此工具返回地址为准",
            parameters={},
            required_params=[]
        )

        return {
            "get_location": location_tool
        }

    async def register_tool(self, tool: ToolDescription):
        """向服务器注册工具"""
        register_request = MCPRequest(
            method="register_tool",
            params={
                "client_id": self.client_id,
                "tool": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "required_params": tool.required_params
                }
            },
            id=str(uuid.uuid4())
        )
        await self.websocket.send(json.dumps(vars(register_request)))
        response = await self.websocket.recv()
        print(f"Tool registration response: {response}")

    async def connect(self):
        """连接到服务器并注册客户端和工具"""
        self.websocket = await websockets.connect(self.server_url)
        
        # 注册客户端
        register_request = MCPRequest(
            method="register",
            params={"client_id": self.client_id},
            id=str(uuid.uuid4())
        )
        await self.websocket.send(json.dumps(vars(register_request)))
        response = await self.websocket.recv()
        print(f"Client registration response: {response}")

        # 注册所有工具
        for tool in self.tools.values():
            await self.register_tool(tool)

    async def get_location(self):
        """获取当前IP的位置信息"""
        url = "https://restapi.amap.com/v3/ip"
        params = {"key": self.amap_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "1" :
                # return data
                print("data",data)
                return f"当前位置信息为：所处国家中国，所处省份{data.get("province")}，所在坐标{data.get("rectangle")}，所在城市{data.get("city")} ,获取到的'city_code'值 为{data.get("adcode")}"
            return "无法获取当前位置信息"
        except Exception as e:
            print(f"获取位置信息失败: {e}")
            return "无法获取当前位置信息"
        
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """处理来自服务器的请求"""
        print("接受内容",request)
        try:
            if request.method == "get_location":
                result = await self.get_location()
                return MCPResponse(result=result, id=request.id)
            else:
                return MCPResponse(
                    result=None,
                    error=f"Unknown method: {request.method}",
                    id=request.id
                )
        except Exception as e:
            return MCPResponse(
                result=None,
                error=str(e),
                id=request.id
            )

    async def start(self):
        """启动客户端"""
        await self.connect()
        try:
            async for message in self.websocket:
                data = json.loads(message)
                request = MCPRequest(**data)
                response = await self.handle_request(request)
                await self.websocket.send(json.dumps(vars(response)))
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed")
        finally:
            if self.websocket:
                await self.websocket.close()

if __name__ == "__main__":
    client = LocalClient()
    asyncio.run(client.start()) 