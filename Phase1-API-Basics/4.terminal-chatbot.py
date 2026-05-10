import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()
api_key = os.environ.get("DEEPSEEK_API_KEY")
print(api_key)
def sendMessage(messages):
    # 使用requests调用 API
    # 构建请求 URL及API端口
    url = "https://api.deepseek.com/chat/completions"
    # 构建请求数据与模型
    data = {
        'model': "deepseek-v4-pro",
        'messages': messages
    }
    # 构建请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

def sendChat(messages):
    # 调用 API
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    return response

# 核心变量：消息历史列表
messages = [
    {"role": "system", "content": "你是一个有用的助手。"} # 可选系统提示
]

while True:
    # 1. 获取用户输入
    user_input = input("你：")
    if user_input == "exit":
        break
    
    # 2. 用户消息加入历史
    messages.append({"role": "user", "content": user_input})
    
    # 3. 把整个 messages 列表发给 API
    response = sendMessage(messages) 
    
    # 4. 获取助手回复（字典方式访问）
    assistant_msg = response['choices'][0]['message']
    
    # 5. 助手回复也要加入历史，否则下次就忘了
    messages.append({"role": "assistant", "content": assistant_msg['content']})
    
    # 6. 打印回复
    print("AI：", assistant_msg['content'])