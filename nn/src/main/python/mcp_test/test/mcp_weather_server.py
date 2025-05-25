# mcp_weather_server.py
from fastapi import FastAPI
from modelcontextprotocol.server import MCPRoute
import httpx

app = FastAPI()
mcp_route = MCPRoute(app, service_name="weather")

@mcp_route.on("get_weather")
async def get_weather(city: str):
    # 模拟调用真实天气 API（例如 OpenWeatherMap）
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # 替换为你的 API Key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["description"]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)