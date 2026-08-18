from __future__ import annotations

from typing import Any

import httpx

from mcp.server import MCPServer

# 直接复用项目已有工具
from app.agent.tools import calculator as local_calculator
from rag.knowledge_base import (
    search_knowledge as local_search_knowledge,
)


# =========================================================
# MCP Server
# =========================================================

mcp = MCPServer(
    "agentic-rag-tools"
)


# =========================================================
# Tool 1：计算器
# 直接复用项目原来的 calculator
# =========================================================

@mcp.tool()
def calculator(
    expression: str,
) -> str:
    """
    安全计算数学表达式。

    例如：
    12 * 8 + 3
    """

    return str(
        local_calculator(
            expression
        )
    )


# =========================================================
# Tool 2：天气
# Open-Meteo：
# 城市 -> 经纬度 -> 当前天气
# =========================================================

WEATHER_CODE_TEXT = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}


@mcp.tool()
async def weather(
    city: str,
) -> dict[str, Any]:
    """
    查询指定城市当前天气。

    例如：
    Tokyo
    Beijing
    Shanghai
    """

    city = city.strip()

    if not city:
        raise ValueError(
            "city 不能为空"
        )

    timeout = httpx.Timeout(
        10.0
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        # -------------------------
        # 1. 城市 -> 经纬度
        # -------------------------

        geo_response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
        )

        geo_response.raise_for_status()

        geo_data = (
            geo_response.json()
        )

        results = (
            geo_data.get("results")
            or []
        )

        if not results:
            raise ValueError(
                f"没有找到城市：{city}"
            )

        location = results[0]

        latitude = location[
            "latitude"
        ]

        longitude = location[
            "longitude"
        ]

        # -------------------------
        # 2. 经纬度 -> 当前天气
        # -------------------------

        weather_response = (
            await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "current": (
                        "temperature_2m,"
                        "apparent_temperature,"
                        "relative_humidity_2m,"
                        "precipitation,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),

                    "timezone":
                        "auto",
                },
            )
        )

        weather_response.raise_for_status()

        weather_data = (
            weather_response.json()
        )

    current = (
        weather_data.get(
            "current",
            {}
        )
    )

    weather_code = int(
        current.get(
            "weather_code",
            -1,
        )
    )

    return {
        "city":
            location.get(
                "name",
                city,
            ),

        "country":
            location.get(
                "country",
                "",
            ),

        "time":
            current.get(
                "time"
            ),

        "weather":
            WEATHER_CODE_TEXT.get(
                weather_code,
                f"天气代码 {weather_code}",
            ),

        "temperature_c":
            current.get(
                "temperature_2m"
            ),

        "apparent_temperature_c":
            current.get(
                "apparent_temperature"
            ),

        "humidity_percent":
            current.get(
                "relative_humidity_2m"
            ),

        "precipitation_mm":
            current.get(
                "precipitation"
            ),

        "wind_speed_kmh":
            current.get(
                "wind_speed_10m"
            ),
    }


# =========================================================
# Tool 3：知识库检索
# 直接复用现有 Agentic RAG
# =========================================================

@mcp.tool()
def search_knowledge(
    query: str,
) -> str:
    """
    查询 Agentic RAG Assistant 本地知识库。

    适合查询项目文档、上传资料以及
    本地 RAG 已经建立索引的内容。
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "query 不能为空"
        )

    return str(
        local_search_knowledge(
            query
        )
    )
