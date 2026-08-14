"""
Tavily 联网搜索工具。

调用链：
Agent
↓
ToolExecutor
↓
tools.py
↓
search_web()
↓
Tavily Search API
"""

from __future__ import annotations

import os

import httpx

from config import config


DEFAULT_ENDPOINT = (
    "https://api.tavily.com/search"
)


def _get_config():
    value = config.get(
        "web_search",
        {},
    )

    if isinstance(value, dict):
        return value

    return {}


def _choose_topic(
    query: str,
) -> str:
    """
    新闻类问题使用 news，
    其他问题使用 general。
    """

    news_keywords = (
        "新闻",
        "最新",
        "最近",
        "近期",
        "今天",
        "今日",
        "本周",
        "刚刚",
    )

    if any(
        word in query
        for word in news_keywords
    ):
        return "news"

    return "general"


def search_web(
    query: str,
) -> str:
    """
    搜索互联网公开信息。

    返回内容会交给大模型整理，
    因此这里只负责：
    搜索 + 整理网页结果。
    """

    if not isinstance(
        query,
        str,
    ):
        return (
            "联网搜索失败："
            "query 必须是字符串。"
        )


    query = query.strip()

    if not query:
        return (
            "联网搜索失败："
            "搜索内容不能为空。"
        )


    # 防止模型生成非常长的搜索词
    query = query[:300]


    web_config = (
        _get_config()
    )


    if not web_config.get(
        "enabled",
        True,
    ):
        return "联网搜索当前已关闭。"


    provider = str(
        web_config.get(
            "provider",
            "tavily",
        )
    ).strip().lower()


    if provider != "tavily":
        return (
            "联网搜索配置错误："
            f"暂不支持 provider={provider}"
        )


    # 优先读取 config.toml，
    # 也支持 TAVILY_API_KEY 环境变量。
    api_key = str(
        web_config.get(
            "api_key",
            "",
        )
        or os.getenv(
            "TAVILY_API_KEY",
            "",
        )
    ).strip()


    if not api_key:
        return (
            "联网搜索不可用："
            "没有配置 Tavily API Key。"
        )


    endpoint = str(
        web_config.get(
            "base_url",
            DEFAULT_ENDPOINT,
        )
    ).strip()


    search_depth = str(
        web_config.get(
            "search_depth",
            "basic",
        )
    ).strip().lower()


    # 当前项目优先节省免费额度。
    if search_depth not in (
        "basic",
        "advanced",
    ):
        search_depth = "basic"


    try:
        max_results = int(
            web_config.get(
                "max_results",
                5,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        max_results = 5


    max_results = max(
        1,
        min(
            max_results,
            10,
        ),
    )


    try:
        timeout = float(
            web_config.get(
                "timeout",
                30.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        timeout = 30.0


    topic = _choose_topic(
        query
    )


    payload = {
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "topic": topic,

        # Agent 自己负责最终总结，
        # Tavily 不需要再生成一份答案。
        "include_answer": False,

        # 避免返回完整网页正文，
        # 防止上下文过大。
        "include_raw_content": False,

        "include_images": False,

        # 显式关闭自动参数，
        # 避免自动把 basic 升为 advanced。
        "auto_parameters": False,
    }


    headers = {
        "Authorization": (
            f"Bearer {api_key}"
        ),
        "Content-Type": (
            "application/json"
        ),
    }


    try:
        response = httpx.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

    except httpx.TimeoutException:
        return (
            "联网搜索失败："
            "Tavily 请求超时。"
        )

    except httpx.HTTPError as error:
        return (
            "联网搜索失败："
            f"{type(error).__name__}"
        )


    if response.status_code == 401:
        return (
            "联网搜索失败："
            "Tavily API Key 无效。"
        )


    if response.status_code == 429:
        return (
            "联网搜索失败："
            "Tavily 请求过于频繁。"
        )


    if response.status_code in (
        432,
        433,
    ):
        return (
            "联网搜索失败："
            "Tavily 当前额度不足。"
        )


    if response.status_code != 200:
        return (
            "联网搜索失败："
            f"HTTP {response.status_code}"
        )


    try:
        data = response.json()

    except ValueError:
        return (
            "联网搜索失败："
            "返回内容不是有效 JSON。"
        )


    results = data.get(
        "results",
        [],
    )


    if not isinstance(
        results,
        list,
    ):
        return (
            "联网搜索失败："
            "搜索结果格式异常。"
        )


    if not results:
        return (
            "联网搜索没有找到"
            "相关网页结果。"
        )


    output = [
        "以下是互联网搜索结果：",
        f"搜索内容：{query}",
        f"搜索类型：{topic}",
        "",
    ]


    valid_count = 0


    for item in results:

        if not isinstance(
            item,
            dict,
        ):
            continue


        title = str(
            item.get(
                "title",
                "",
            )
        ).strip()


        url = str(
            item.get(
                "url",
                "",
            )
        ).strip()


        content = str(
            item.get(
                "content",
                "",
            )
        ).strip()


        score = item.get(
            "score"
        )


        # 控制每个网页片段长度
        if len(content) > 700:
            content = (
                content[:700]
                + "..."
            )


        valid_count += 1


        output.append(
            f"[网页 {valid_count}]"
        )


        if title:
            output.append(
                f"标题：{title}"
            )


        if content:
            output.append(
                f"摘要：{content}"
            )


        if url:
            output.append(
                f"链接：{url}"
            )


        if score is not None:
            output.append(
                f"搜索相关分数：{score}"
            )


        output.append("")


    if valid_count == 0:
        return (
            "联网搜索没有找到"
            "格式有效的结果。"
        )


    response_time = data.get(
        "response_time"
    )

    if response_time:
        output.append(
            "Tavily 搜索耗时："
            f"{response_time} 秒"
        )


    return "\n".join(
        output
    ).strip()
