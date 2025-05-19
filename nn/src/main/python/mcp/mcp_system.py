# server_demo.py
from mcp.server.fastmcp import FastMCP
import platform

# 创建一个MCP服务器
mcp = FastMCP("本机信息查询")

# 添加一个查询本机信息的工具
@mcp.tool()
def get_system_info():
    """获取本机系统信息"""
    return {
        "system": platform.system(),
        "node_name": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }

if __name__ == "__main__":
    # 运行MCP Server，采用标准输入输出传输方式
    mcp.run(transport='stdio')