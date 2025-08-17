
import websockets
from mcp.utils.base_protocol import MCPRequest, MCPResponse, ToolDescription
import json
from mcp.utils import config
from typing import Dict
import asyncio
import uuid

class QuickApi:
    def __init__(self, server_url: str = config.base_url):
        self.server_url = server_url
    
    async def call_tools(self,websocket,**params):
      
        mcp_func = params.pop("mcp_func")
        """获取当前所有的tool"""
        register_request = MCPRequest(
            method="call_tool",
            params={"tool_name":mcp_func,"tool_params":params},
            id=str(uuid.uuid4())
        )
        await websocket.send(json.dumps(vars(register_request)))
        # response = "await websocket.recv()"
        response = await websocket.recv()
        print("response",response,json.loads(response))
        return response

    async def get_tools(self,websocket,**params):
        """获取当前所有的tool"""
        register_request = MCPRequest(
            method="list_tools",
            params={
            },
            id=str(uuid.uuid4())
        )
        await websocket.send(json.dumps(vars(register_request)))
        response = await websocket.recv()
        return response
        # print(f"Tool registration response: {response}")

    async def connect(self):
        """连接到服务器并注册客户端和工具"""
        return await websockets.connect(self.server_url)

   
    async def run_func(self,func,**paramers):
        """启动客户端"""
        websocket = await self.connect()
        try:
            return await func(websocket,**paramers)
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed")
        finally:
            if websocket:
                await websocket.close()

    

if __name__ == "__main__":
    qa = QuickApi()
    asyncio.run(qa.start())


    