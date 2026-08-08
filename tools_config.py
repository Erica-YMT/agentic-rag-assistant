tools = [
{
"type":"function",
"function":{
"name":"search_knowledge",
"description":
"""
查询知识库。
调用时，请将用户问题改写成适合搜索的关键词。
例如：
用户:
登录那个问题怎么解决？
query:
登录失败 JWT token认证异常 排查方案
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
搜索互联网获取实时或外部公开信息。

以下情况优先使用：
- 今天、最新、近期、当前发生的事情
- 新闻、政策、版本、价格等可能变化的信息
- 用户明确要求联网搜索、网上查找、搜索网页
- 本地知识库中没有，但互联网可能存在的信息

不要用它代替本地知识库：
当前项目代码、项目文档、本地资料应优先使用 search_knowledge。
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
}
]