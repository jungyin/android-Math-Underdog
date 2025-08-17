import asyncio
import json
import uuid
import websockets
import numpy as np
from ..utils.base_protocol import MCPRequest, MCPResponse, ToolDescription
import requests
from typing import Dict
from ..utils import config
from infer.qwen.base_infer import BaseMoelRun as lmBaseModel
from infer.qwen import onnx_infer as qwen0_5b_onnx,openvino_infer as qwen0_5b_openvino ,source_infer as qwen0_5b_torch
from tokenizers import Tokenizer
import time



class CoreLLMClient:
    def __init__(self, server_url: str = config.base_url, amap_key: str = "c49485098e88a45c418addbbfe4de800"):
        self.server_url = server_url
        self.client_id = f"local_client_{uuid.uuid4().hex[:8]}"
        self.amap_key = amap_key
        self.websocket = None
        self.tools = self.create_tools()
        
        self.llm_model :lmBaseModel = None

        # 注册所有的服务模块
        self.llm_model_list = {
                'qwen0_5b_onnx':{"model":qwen0_5b_onnx,"model_path" :'./assets/qewn2/'},
                # 'qwen0_5b_openvino':{"model":qwen0_5b_openvino,"model_path" :'./assets/qewn2/'},
                'qwen0_5b_openvino':{"model":qwen0_5b_openvino,"model_path" :'D:/code/transformer_models/models--Qwen--Qwen2.5-3B-Instruct/'},
                'qwen0_5b_torch':{"model":qwen0_5b_torch,"model_path" :'D:/code/transformer_models/models--Qwen--Qwen2.5-3B-Instruct/'},
                # 'qwen0_5b_torch':{"model":qwen0_5b_torch,"model_path" :'./assets/qewn2/'},
            }
        self.local_token = None

        self.select_llm_model("qwen0_5b_torch")

    def select_llm_model(self,modelkeys):
        """
        选中指定模型
        @params modelkeys: 选中的模型
        return  msg:配置返回字符
        """

        if(modelkeys in self.llm_model_list.keys()):
            m_path = self.llm_model_list[modelkeys]['model_path']
            llm_model = self.llm_model_list[modelkeys]['model']
            self.llm_model = llm_model.QwenMoelRun(m_path)
            self.local_token = Tokenizer.from_file(m_path+"/tokenizer.json")
            return ""
        else:
            return "error cannot select this llm model"
        
    def create_tools(self) -> Dict[str, ToolDescription]:
        """创建工具描述"""
        location_tool = ToolDescription(
            name="local_llm",
            description="local_llm",
            parameters={},
            required_params=[]
        )

        return {
            "location_tool": location_tool
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
        
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """处理来自服务器的请求"""
        print("接受内容",request)
        # try:
        if True:
            if request.method == "local_llm":
                params = request.params
                func_name = params["method"]
                if func_name == "stopGenerate":
                    await self.llm_model.stopGenerate()
                    return MCPResponse(result="work end", id=request.id)

                    
                elif func_name == "selectModels":
                    
                    rmessage = self.select_llm_model(params["model"])

                    if(rmessage == ""):
                        return MCPResponse(result="work surcess", id=request.id)
                    else:
                        return MCPResponse(
                            result=None,
                            error=rmessage,
                            id=request.id
                        )
                    
                elif func_name == "generate":
                    print("让我看看找到啥了1",params)
                    build_text = self.llm_model.build_input_text(params["messages"])
                    print(build_text)
                    encoding = self.local_token.encode_batch(
                            [build_text],
                            add_special_tokens=True,
                            is_pretokenized=False,
                        )
                        
                    input_ids = encoding[0].ids
                    input_ids = np.array(input_ids,np.int64)

                    output = self.llm_model.generate(input_ids,None,self.local_token)
                    output  = self.local_token.decode(output,skip_special_tokens=True)
                    return MCPResponse(result=output, id=request.id)
                else:
                    return MCPResponse(
                        result=None,
                        error=f"Unknown method: {request.method}",
                        id=request.id
                    )
                
            else:
                return MCPResponse(
                    result=None,
                    error=f"Unknown method: {request.method}",
                    id=request.id
                )
        # except Exception as e:
        #     return MCPResponse(
        #         result=None,
        #         error=str(e),
        #         id=request.id
        #     )

    async def start(self):
        """启动客户端"""
        await self.connect()
        try:
            async for message in self.websocket:
                data = json.loads(message)
                request = MCPRequest(**data)
                response = await self.handle_request(request)
                print("待返回内容",response)
                await self.websocket.send(json.dumps(vars(response)))
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed")
        finally:
            if self.websocket:
                await self.websocket.close()

if __name__ == "__main__":
    client = CoreLLMClient()
    asyncio.run(client.start()) 