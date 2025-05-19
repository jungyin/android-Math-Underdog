import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any
import os
import sys
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from openai import AsyncAzureOpenAI

# Configure logging
logging.basicConfig(
   level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
python_executable = sys.executable
   
   # 获取虚拟环境的 site-packages 路径
venv_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_executable)), 'Lib', 'site-packages')


class Configuration:

   def __init__(self) -> None:
       self.load_env()
       self.api_key = os.getenv("AZURE_OPENAI_KEY")

   @staticmethod
   def load_env() -> None:
       load_dotenv(dotenv_path='.env.dev')

   @staticmethod
   def load_config(file_path: str) -> dict[str, Any]:
       with open(file_path, "r") as f:
           return json.load(f)

   @property
   def llm_api_key(self) -> str:
       if not self.api_key:
           raise ValueError("LLM_API_KEY not found in environment variables")
       return self.api_key


class Server:

   def __init__(self, name: str, config: dict[str, Any]) -> None:
       self.name: str = name
       self.config: dict[str, Any] = config
       self.stdio_context: Any | None = None
       self.session: ClientSession | None = None
       self._cleanup_lock: asyncio.Lock = asyncio.Lock()
       self.exit_stack: AsyncExitStack = AsyncExitStack()

   async def initialize(self) -> None:
       command = self.config.get("command")
       if command is None:
           raise ValueError("The command must be a valid string and cannot be None.")

       # 检查是否是 stdio 模式
       is_stdio = any(arg == "stdio" for arg in self.config.get("args", []))
       
       if is_stdio:
           # 处理 stdio 连接
           server_params = StdioServerParameters(
               command=command,
               args=self.config["args"],
               env={"PYTHONPATH": venv_site_packages, **os.environ, **self.config.get("env", {})}
           )
           
           try:
               logging.info(f"Initializing stdio connection for server {self.name}")
               stdio_transport = await self.exit_stack.enter_async_context(
                   stdio_client(server_params)
               )
               read, write = stdio_transport
               session = await self.exit_stack.enter_async_context(
                   ClientSession(read, write)
               )
               await session.initialize()
               self.session = session
               logging.info(f"Successfully initialized stdio connection for server {self.name}")
           except Exception as e:
               logging.error(f"Error initializing stdio server {self.name}: {e}")
               await self.cleanup()
               raise
           return

       # 如果不是 stdio 模式，则尝试 SSE 连接
       if "--transport" in self.config.get("args", []) and "sse" in self.config.get("args", []):
           try:
               port = 8000  # 默认端口
               # 从参数中获取端口
               if "--port" in self.config["args"]:
                   port_index = self.config["args"].index("--port") + 1
                   if port_index < len(self.config["args"]):
                       port = int(self.config["args"][port_index])
               
               # 修改 SSE 端点路径
               base_url = f"http://localhost:{port}/sse"
               logging.info(f"Connecting to SSE endpoint: {base_url}")
               
               # 添加重试逻辑
               max_retries = 3
               retry_delay = 2.0
               
               for attempt in range(max_retries):
                   try:
                       sse_transport = await self.exit_stack.enter_async_context(
                           sse_client(base_url)
                       )
                       read, write = sse_transport
                       session = await self.exit_stack.enter_async_context(
                           ClientSession(read, write)
                       )
                       await session.initialize()
                       self.session = session
                       logging.info("Successfully connected to SSE server")
                       return
                   except Exception as e:
                       if attempt < max_retries - 1:
                           logging.warning(f"Connection attempt {attempt + 1} failed: {e}")
                           await asyncio.sleep(retry_delay)
                       else:
                           raise
                           
           except Exception as e:
               logging.error(f"Error initializing SSE server {self.name}: {e}")
               await self.cleanup()
               raise

   async def list_tools(self) -> list[Any]:
       if not self.session:
           raise RuntimeError(f"Server {self.name} not initialized")

       tools_response = await self.session.list_tools()
       tools = []

       for item in tools_response:
           if isinstance(item, tuple) and item[0] == "tools":
               for tool in item[1]:
                   tools.append(Tool(tool.name, tool.description, tool.inputSchema))

       return tools

   async def execute_tool(
       self,
       tool_name: str,
       arguments: dict[str, Any],
       retries: int = 2,
       delay: float = 1.0,
   ) -> Any:
       
       if not self.session:
           raise RuntimeError(f"Server {self.name} not initialized")

       attempt = 0
       while attempt < retries:
           try:
               logging.info(f"Executing {tool_name}...")
               result = await self.session.call_tool(tool_name, arguments)

               return result

           except Exception as e:
               attempt += 1
               logging.warning(
                   f"Error executing tool: {e}. Attempt {attempt} of {retries}."
               )
               if attempt < retries:
                   logging.info(f"Retrying in {delay} seconds...")
                   await asyncio.sleep(delay)
               else:
                   logging.error("Max retries reached. Failing.")
                   raise

   async def cleanup(self) -> None:
       async with self._cleanup_lock:
           try:
               await self.exit_stack.aclose()
               self.session = None
               self.stdio_context = None
           except Exception as e:
               logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:

   def __init__(
       self, name: str, description: str, input_schema: dict[str, Any]
   ) -> None:
       self.name: str = name
       self.description: str = description
       self.input_schema: dict[str, Any] = input_schema

   def format_for_llm(self) -> str:
    
       args_desc = []
       if "properties" in self.input_schema:
           for param_name, param_info in self.input_schema["properties"].items():
               arg_desc = (
                   f"- {param_name}: {param_info.get('description', 'No description')}"
               )
               if param_name in self.input_schema.get("required", []):
                   arg_desc += " (required)"
               args_desc.append(arg_desc)

       return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
   def __init__(self, api_key: str) -> None:
       self.api_key: str = api_key
       self.client = AsyncAzureOpenAI(
           api_key=os.environ.get("AZURE_OPENAI_KEY"),
           api_version=os.environ.get("AZURE_OPENAI_VERSION"),
           azure_endpoint=os.environ.get("AZURE_OPENAI_BASE")
       )

   async def get_response(self, messages: list[dict[str, str]]) -> str:
       formatted_messages = [
           {"role": message["role"], "content": message["content"]}
           for message in messages
       ]

       response =  await self.client.chat.completions.create(
               model=os.environ.get("AZURE_OPENAI_MODEL"),
               messages=formatted_messages,
               temperature=0,
               max_tokens=10000,
               stream=False,
           )
       
       return response.choices[0].message.content


class ChatSession:
   def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
       self.servers: list[Server] = servers
       self.llm_client: LLMClient = llm_client

   async def cleanup_servers(self) -> None:
       cleanup_tasks = []
       for server in self.servers:
           cleanup_tasks.append(asyncio.create_task(server.cleanup()))

       if cleanup_tasks:
           try:
               await asyncio.gather(*cleanup_tasks, return_exceptions=True)
           except Exception as e:
               logging.warning(f"Warning during final cleanup: {e}")

   async def process_llm_response(self, llm_response: str) -> str:

       import json

       try:
           tool_call = json.loads(llm_response)
           if "tool" in tool_call and "arguments" in tool_call:
               logging.info(f"Executing tool: {tool_call['tool']}")
               logging.info(f"With arguments: {tool_call['arguments']}")
         
               for server in self.servers:
                   tools = await server.list_tools()
                   if any(tool.name == tool_call["tool"] for tool in tools):
                       try:
                           result = await server.execute_tool(
                               tool_call["tool"], tool_call["arguments"]
                           )

                           if isinstance(result, dict) and "progress" in result:
                               progress = result["progress"]
                               total = result["total"]
                               percentage = (progress / total) * 100
                               logging.info(
                                   f"Progress: {progress}/{total} "
                                   f"({percentage:.1f}%)"
                               )

                           return f"Tool execution result: {result}"
                       except Exception as e:
                           error_msg = f"Error executing tool: {str(e)}"
                           logging.error(error_msg)
                           return error_msg

               return f"No server found with tool: {tool_call['tool']}"
           return llm_response
       except json.JSONDecodeError:
           return llm_response

   async def start(self) -> None:
       try:
           for server in self.servers:
               try:
                   await server.initialize()
               except Exception as e:
                   logging.error(f"Failed to initialize server: {e}")
                   await self.cleanup_servers()
                   return

           all_tools = []
           for server in self.servers:
               tools = await server.list_tools()
               all_tools.extend(tools)

           tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

           system_message = (
               "You are a helpful assistant with access to these tools:\n\n"
               f"{tools_description}\n"
               "Choose the appropriate tool based on the user's question. "
               "If no tool is needed, reply directly.\n\n"
               "IMPORTANT: When you need to use a tool, you must ONLY respond with "
               "the exact JSON object format below, nothing else:\n"
               "{\n"
               '    "tool": "tool-name",\n'
               '    "arguments": {\n'
               '        "argument-name": "value"\n'
               "    }\n"
               "}\n\n"
               "After receiving a tool's response:\n"
               "1. Transform the raw data into a natural, conversational response\n"
               "2. Keep responses concise but informative\n"
               "3. Focus on the most relevant information\n"
               "4. Use appropriate context from the user's question\n"
               "5. Avoid simply repeating the raw data\n\n"
               "Please use only the tools that are explicitly defined above."
           )

           messages = [{"role": "system", "content": system_message}]

           while True:
               try:
                   user_input = input("请输入你的问题: ").strip().lower()
                   if user_input in ["quit", "exit"]:
                       logging.info("\nExiting...")
                       break

                   messages.append({"role": "user", "content": user_input})

                   llm_response = await self.llm_client.get_response(messages)
                   logging.info("\nAssistant: %s", llm_response)

                   result = await self.process_llm_response(llm_response)

                   if result != llm_response:
                       messages.append({"role": "assistant", "content": llm_response})
                       messages.append({"role": "system", "content": result})

                       final_response = await self.llm_client.get_response(messages)
                       logging.info("\nFinal response: %s", final_response)
                       messages.append(
                           {"role": "assistant", "content": final_response}
                       )
                   else:
                       messages.append({"role": "assistant", "content": llm_response})

               except KeyboardInterrupt:
                   logging.info("\nExiting...")
                   break

       finally:
           await self.cleanup_servers()


async def main() -> None:
   config = Configuration()
   server_config = config.load_config("servers_config.json")
   servers = [
       Server(name, srv_config)
       for name, srv_config in server_config["mcpServers"].items()
   ]
   llm_client = LLMClient(config.llm_api_key)
   chat_session = ChatSession(servers, llm_client)
   await chat_session.start()


if __name__ == "__main__":
   asyncio.run(main())
