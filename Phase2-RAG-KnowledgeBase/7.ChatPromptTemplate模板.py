
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompt_values import ChatPromptValue
# ==============================================
# 知识点1: ChatPromptTemplate创建
# ==============================================
print("\n" + "=" * 60)
print("知识点1: ChatPromptTemplate创建")
# 方式1: 使用ChatPromptTemplate直接创建
chat_prompt1 = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("你是一个{role}。"),
        HumanMessagePromptTemplate.from_template("{question}")
    ],
    input_variables=["role", "question"]
)

# 方式2: 使用ChatPromptTemplate.from_messages创建
chat_prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。"),
    ("human", "{question}")
])


# ==============================================
# 知识点2: ChatPromptTemplate四种填充方法
# format: 传入变量值返回字符串类型
# invoke: 传入字典返回ChatPromptValue
# format_prompt: 传入变量值返回ChatPromptValue
# format_messages: 传入变量值返回消息构成的列表
# ==============================================
print("\n" + "=" * 60)
print("知识点2: ChatPromptTemplate四种填充方法")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{field}专家。"),
    ("human", "{user_input}")
])

# 1. format - 返回字符串
result_format = chat_prompt.format(field="计算机科学", user_input="什么是机器学习？")
print("1. format方法 - 类型:", type(result_format))

# 2. invoke - 返回ChatPromptValue
result_invoke = chat_prompt.invoke({"field": "计算机科学", "user_input": "什么是机器学习？"})
print("2. invoke方法 - 类型:", type(result_invoke))

# 3. format_prompt - 返回ChatPromptValue
result_format_prompt = chat_prompt.format_prompt(field="计算机科学", user_input="什么是机器学习？")
print("3. format_prompt方法 - 类型:", type(result_format_prompt))

# 4. format_messages - 返回消息列表
result_format_messages = chat_prompt.format_messages(field="计算机科学", user_input="什么是机器学习？")
print("4. format_messages方法 - 类型:", type(result_format_messages), "- 结果:", result_format_messages)

# ==============================================
# 知识点3: ChatPromptValue类型的转换
# to_messages: 转为消息列表
# to_string: 转为字符串
# ==============================================
print("\n" + "=" * 60)
print("知识点3: ChatPromptValue类型的转换")
# 转为消息列表
messages = result_invoke.to_messages()
print("to_messages() - 类型:", type(messages), "- 结果:", messages)

# 转为字符串
text = result_invoke.to_string()
print("to_string() - 类型:", type(text), "- 结果:", text)

# ==============================================
# 知识点4: MessagePlaceholder的使用
# 当消息类型和个数不确定时使用
# ==============================================
print("\n" + "=" * 60)
print("知识点4: MessagePlaceholder的使用")
# 创建带有MessagePlaceholder的模板
prompt_with_placeholder = ChatPromptTemplate.from_messages([
    ("system", "你是一个对话助手，根据历史对话回答问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 模拟不同长度的历史对话
history1 = [
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么我可以帮助你的吗？")
]

history2 = [
    HumanMessage(content="什么是AI？"),
    AIMessage(content="人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"),
    HumanMessage(content="AI有哪些应用？"),
    AIMessage(content="AI应用广泛，包括自动驾驶、语音识别、推荐系统等。")
]

print("历史对话1结果:")
result1 = prompt_with_placeholder.format_messages(history=history1, question="你是谁？")
for msg in result1:
    print(f"  {msg.type}: {msg.content}")

print("\n历史对话2结果:")
result2 = prompt_with_placeholder.format_messages(history=history2, question="那机器学习呢？")
for msg in result2:
    print(f"  {msg.type}: {msg.content}")
