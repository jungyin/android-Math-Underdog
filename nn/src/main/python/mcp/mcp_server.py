# # client_demo.py
# from mcp.client.stdio import stdio_client
# from mcp import ClientSession, StdioServerParameters, types
# import asyncio

# # Client会使用这里的配置来启动本地MCP Server
# server_params = StdioServerParameters(
#     command="python",
#     args=["./mcp/mcp_system.py"],
#     env=None
# )

# async def main():
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write, sampling_callback=None) as session:
#             await session.initialize()
#             print('\n正在调用工具...')
#             result = await session.call_tool("get_system_info", {})
#             print(result.content)

# # 执行异步主函数
# asyncio.run(main())


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/mcp_endpoint', methods=['POST'])
def mcp_service():
    data_received = request.json
    # 这里处理接收到的数据
    response_data = {"status": "success", "data": data_received}
    return jsonify(response_data)

if __name__ == '__main__':
    # 你可以在这里指定想要监听的端口号
    port = 8000  # 假设你想在8000端口上运行
    app.run(host='0.0.0.0', port=port)