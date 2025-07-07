import asyncio
import json
import uuid
import websockets
from ..utils.base_protocol import MCPRequest, MCPResponse, ToolDescription
import requests
from typing import Dict
from ..utils import config

class WeatherClient:
    def __init__(self, server_url: str = config.base_url, amap_key: str = "c49485098e88a45c418addbbfe4de800"):
        self.server_url = server_url
        self.client_id = f"weather_client_{uuid.uuid4().hex[:8]}"
        self.amap_key = amap_key
        self.websocket = None
        self.tools = self.create_tools()

    def create_tools(self) -> Dict[str, ToolDescription]:
        """创建工具描述"""
        weather_tool = ToolDescription(
            name="get_weather",
            description="获取指定城市的天气信息。使用该工具前,需确定当前用户输入中有位置信息,如果没有，则需要先进行一次定位,如果用户未指定时间，则默认天气预报类型为：base（实时）",
            parameters={
                "city": {
                    "type": "string",
                    "description": "城市编码或名称"
                },
                "extensions": {
                    "type": "string",
                    "description": "天气预报类型：base（实时）或 all（预报）",
                    "default": "base"
                }
            },
            required_params=["city"]
        )
        return {
            "get_weather": weather_tool,
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


    async def get_weather(self, city: str, extensions: str = "base"):
        """获取天气信息"""
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "key": self.amap_key,
            "city": city,
            "extensions": extensions,
            "output": "JSON"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            print("data",data,type(data))
          
            if data.get("status") == "1":
                ds = data.get("lives")
                if(len(ds)>0 and len(ds[0]) > 0):
                    return f"当前位置下获取到的天气信息为：湿度${ds[0][0].get("humidity_float")},风向${ds[0][0].get("winddirection")},风力为${ds[0][0].get("windpower")}级,温度为${ds[0][0].get("temperature_float")},天气为${ds[0][0].get("weather")}"
            return "天气获取失败"
        except Exception as e:
            print(f"获取天气信息失败: {e}",params)
            return "天气获取失败"

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """处理来自服务器的请求"""
        try:
    
            if request.method == "get_weather":
                result = await self.get_weather(
                    request.params.get("city"),
                    request.params.get("extensions", "base")
                )
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
    client = WeatherClient()
    data = asyncio.run(client.get_weather("1000"))
    asyncio.run(client.start()) 