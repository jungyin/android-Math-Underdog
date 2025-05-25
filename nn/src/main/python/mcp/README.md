# MCP (Model Context Protocol) 示例

这是一个简单的MCP（Model Context Protocol）示例项目，展示了如何实现一个基于WebSocket的MCP服务器和客户端。该示例包含了一个天气预报工具的实现。

## 项目结构

```
mcp/
├── server/
│   └── mcp_server.py    # MCP服务器实现
├── client/
│   └── weather_client.py # 天气预报客户端实现
├── utils/
│   └── base_protocol.py # 基础协议定义
├── requirements.txt     # 项目依赖
└── README.md           # 项目说明文档
```

## 功能特性

- MCP服务器支持动态工具注册和发现
- 支持获取当前IP地址的位置信息
- 支持查询指定城市的天气信息
- 基于WebSocket的实时通信
- 支持异步操作

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动MCP服务器：

```bash
python -m mcp.server.mcp_server
```

2. 启动天气预报客户端：

```bash
python -m mcp.client.weather_client
```

## 可用工具

### 1. 获取位置信息 (get_location)
- 描述：获取当前IP地址的位置信息
- 参数：无需参数

### 2. 获取天气信息 (get_weather)
- 描述：获取指定城市的天气信息
- 必需参数：
  - city: 城市编码或名称
- 可选参数：
  - extensions: 天气预报类型（base: 实时天气，all: 天气预报）

## 示例查询

要查询可用工具列表：
```python
request = MCPRequest(
    method="list_tools",
    params={},
    id="unique_id"
)
```

要查询天气信息：
```python
request = MCPRequest(
    method="get_weather",
    params={
        "city": "110000",  # 北京的城市编码
        "extensions": "base"
    },
    id="unique_id"
)
```

## 注意事项

- 使用前请确保已配置正确的高德地图API密钥
- 服务器默认运行在 localhost:8765
- 所有网络请求都有10秒的超时限制 