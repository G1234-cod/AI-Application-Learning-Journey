#导入库
import os #设置本地环境
from dotenv import load_dotenv #用于获取本地文件，如.env
from langchain_openai import ChatOpenAI  # 注意：从 langchain_openai 导入

# 加载 .env 文件中的环境变量
load_dotenv()
# 从文档中获取 API 密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

#获取对话模型
chat_model = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model=model_name
)

# 调用模型
response = chat_model.invoke("2026年5月14日是星期几？") #返回的是一个AIMessage对象
print(type(response))
print(response.content)