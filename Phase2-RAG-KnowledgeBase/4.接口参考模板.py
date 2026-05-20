# 导入库
import os  # 设置本地环境
import asyncio  # 异步编程支持
from dotenv import load_dotenv  # 用于获取本地文件，如.env
from langchain_openai import ChatOpenAI  # 注意：从 langchain_openai 导入

load_dotenv()
    
# 从环境变量中获取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")
    
# 初始化并返回聊天模型
chat_model = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model=model_name
)

# ==========================================
# 一、同步接口示例
# ==========================================
def invoke_demo(chat_model: ChatOpenAI):
    """
    invoke —— 单次同步调用，等待完整结果返回
    适用场景：
        - 需要获取完整回复后再进行后续处理
        - 简单的问答场景
        - 对响应速度要求不高的场景
    返回类型：
        AIMessage 对象，包含完整的回复内容
    """
    print("=== invoke (同步单次调用) ===")
    result = chat_model.invoke("用一句话介绍洛阳理工学院")
    print(result.content)

def stream_demo(chat_model: ChatOpenAI):
    """
    stream —— 同步流式输出，边生成边返回（打字机效果）
    适用场景：
        - 需要实时显示回复内容
        - 长文本生成场景
        - 提升用户体验，避免长时间等待
    返回类型：
        生成器，逐 token 返回 AIMessage 对象
    示例：
        for chunk in chat_model.stream("问题内容"):
            print(chunk.content, end="", flush=True)
    """
    print("=== stream (同步流式输出) ===")
    print("输入：用一句话介绍洛阳理工学院")
    for chunk in chat_model.stream("用一句话介绍洛阳理工学院"):
        # chunk 也是 AIMessage，逐 token 返回
        print(chunk.content, end="", flush=True)


def batch_demo(chat_model: ChatOpenAI):
    """
    batch —— 批量并发处理多个 prompt
    适用场景：
        - 需要同时处理多个独立问题
        - 批量生成内容
        - 提高处理效率，减少网络往返次数
    返回类型：
        列表，包含每个 prompt 的 AIMessage 结果
    """
    print("=== batch (批量并发处理) ===")
    prompts = [
        "用一句话介绍洛阳",
        "用一句话介绍郑州",
        "用一句话介绍开封"
    ]
    results = chat_model.batch(prompts)
    for i, res in enumerate(results):
        print(f"结果 {i+1}: {res.content}")


# ==========================================
# 二、异步接口示例（需要在 async 函数中运行）
# ==========================================
async def async_invoke_demo(chat_model: ChatOpenAI):
    """
    ainvoke —— 异步单次调用
    适用场景：
        - 在异步应用中使用（如 FastAPI、Starlette）
        - 需要并发执行多个独立的 LLM 调用
        - 不阻塞事件循环
    返回类型：
        协程对象，await 后返回 AIMessage
    """
    print("=== ainvoke (异步单次调用) ===")
    res = await chat_model.ainvoke("用一句话介绍洛阳理工学院")
    print(res.content)

async def async_stream_demo(chat_model: ChatOpenAI):
    """
    astream —— 异步流式输出
    适用场景：
        - 异步应用中的实时响应场景
        - WebSocket 实时推送
        - 异步框架中的流式响应
    返回类型：
        异步生成器，逐 token 返回 AIMessage
    """
    print("=== astream (异步流式输出) ===")
    async for chunk in chat_model.astream("用一句话介绍洛阳理工学院"):
        print(chunk.content, end="", flush=True)

async def async_batch_demo(chat_model: ChatOpenAI):
    """
    abatch —— 异步批量处理
    适用场景：
        - 异步应用中的批量任务
        - 需要高并发处理多个请求
        - 资源密集型批量任务
    返回类型：
        协程对象，await 后返回 AIMessage 列表
    示例：
        results = await chat_model.abatch(["问题1", "问题2"])
        for res in results:
            print(res.content)
    """
    print("=== abatch (异步批量处理) ===")
    prompts = [
        "用一句话介绍洛阳",
        "用一句话介绍郑州"
    ]
    results = await chat_model.abatch(prompts)
    for i, res in enumerate(results):
        print(res.content)


async def async_demo(chat_model: ChatOpenAI):
    """
    异步演示主函数 —— 包含所有异步接口示例
    """
    await async_invoke_demo(chat_model)
    await async_stream_demo(chat_model)
    await async_batch_demo(chat_model)


#选择性调用各个功能
# 1. 同步单次调用
# invoke_demo(chat_model)

# 2. 同步流式输出
# stream_demo(chat_model)
        
# 3. 同步批量处理
# batch_demo(chat_model)
        
# 4. 异步接口演示（需要 asyncio.run 运行）
asyncio.run(async_demo(chat_model))
        
