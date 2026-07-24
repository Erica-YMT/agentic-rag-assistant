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
]