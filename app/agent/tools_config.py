tools = [
{
"type":"function",
"function":{
"name":"search_knowledge",
"description":
"""
查询本地知识库、本地文档、项目资料和用户上传资料。

以下情况优先调用本工具：
- 用户询问当前项目、代码、文档或本地资料
- 用户询问一个不熟悉的专有名称、项目名称或内部资料内容
- 用户的问题可能存在于用户上传的文件中
- 仅凭模型自身知识无法确定，而本地知识库可能有答案

特别注意：
- 不要因为模型自己不知道某个名称，就直接调用 search_web
- 应先使用 search_knowledge 检查本地资料
- 只有实际检索知识库后仍然资料不足，才考虑 search_web

调用时，请把用户问题改写成适合知识库检索的关键词。
""",
"parameters":{
"type":"object",
"properties":{
"query":{
"type":"string",
"description":"用户的问题"
}
},
"required":[
"query"
]
}
}
},


{
"type":"function",
"function":{
"name":"calculator",
"description":
"计算数学表达式",
"parameters":{
"type":"object",
"properties":{
"expression":{
"type":"string",
"description":
"""
执行数学计算。
只用于：
- 加减乘除
- 数学表达式
- 数字计算
不要用于：
- 逻辑分析
- 文字处理
- 单位解释
"""
}
},
"required":[
"expression"
]
}
}
}
,
{
"type":"function",
"function":{
"name":"search_web",
"description":
"""
搜索互联网，获取实时或外部公开信息。

以下情况优先调用：
- 今天、最新、近期、实时发生的事情
- 新闻、政策、版本、价格等可能变化的信息
- 用户明确要求联网搜索、网上查找、搜索网页
- 已经实际查询本地知识库，但本地资料不足，而互联网可能存在答案

不要用本工具代替本地知识库：
当前项目、代码、本地文档、用户上传资料，应优先使用 search_knowledge。

对于模型不熟悉的专有名称，不能仅凭“模型不知道”就判断本地没有资料。
应先实际调用 search_knowledge。
""",
"parameters":{
"type":"object",
"properties":{
"query":{
"type":"string",
"description":
"""
适合互联网搜索的关键词或简短问题。
保留关键实体、时间范围和主题。
"""
}
},
"required":[
"query"
]
}
}
},

    {
        "type": "function",
        "function": {
            "name": "mcp_filesystem",
            "description": (
                "通过 Filesystem MCP 访问当前项目文件。"
                "用户要求读取项目文件、列目录、"
                "查找文件时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "read",
                            "list",
                            "search",
                        ],
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "相对于项目根目录的路径"
                        ),
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "search 时使用的搜索模式"
                        ),
                    },
                },
                "required": [
                    "action",
                    "path",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_hot_repositories",
            "description": (
                "通过 GitHub 官方 MCP "
                "查询近期热门 GitHub 仓库。"
                "用户询问近期、本周、本月热门"
                "AI、Agent、RAG、MCP 等"
                "开源项目时优先使用。"
                "结果包含项目说明、Star、语言等信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": (
                            "关键词，例如 AI Agent、RAG、MCP"
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 365,
                        "description": (
                            "最近多少天创建的仓库"
                        ),
                    },
                    "min_stars": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "最低 Star 数"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": (
                            "最多返回多少个仓库"
                        ),
                    },
                },
                "required": [
                    "keyword",
                ],
            },
        },
    },
]