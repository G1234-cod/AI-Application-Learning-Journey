# ==============================================
# Chains 入门学习示例
# 参考 8.返回值类型.py 风格编写
# ==============================================

# 导入必要的库
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

# 初始化聊天模型
llm = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,
    model=model_name
)

# ==============================================
# 方法1: 基础链 - Prompt + LLM + Parser
# 知识点: 使用|操作符串联组件
# ==============================================
def demo_basic_chain():
    """演示最基础的链"""
    print("\n" + "="*50)
    print("【方法1】基础链 (Basic Chain)")
    print("-"*40)
    
    # 创建提示词模板
    prompt = PromptTemplate.from_template("请用一句话介绍{topic}。")
    
    # 创建链: Prompt -> LLM -> Parser
    chain = prompt | llm | StrOutputParser()
    
    # 执行
    result = chain.invoke({"topic": "人工智能"})
    
    # 输出结果
    print(f"提示词: {prompt.format(topic='人工智能')}")
    print(f"结果类型: {type(result)}")
    print(f"结果内容:\n{result}")

# ==============================================
# 方法2: 顺序链 - 单输出接力
# 知识点: 前一个的输出作为下一个的输入
# ==============================================
def demo_sequential_chain():
    """演示顺序链 - 单输出接力"""
    print("\n" + "="*50)
    print("【方法2】顺序链 (Sequential Chain)")
    print("-"*40)
    
    # 步骤1: 翻译
    translate_prompt = PromptTemplate.from_template(
        "请将以下中文翻译成英文：{text}"
    )
    translate_chain = translate_prompt | llm | StrOutputParser()
    
    # 步骤2: 总结
    summarize_prompt = PromptTemplate.from_template(
        "请用一句话总结以下英文内容：{text}"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    
    # 执行
    input_text = "人工智能正在改变我们的生活方式，从自动驾驶到智能助手。"
    translated = translate_chain.invoke({"text": input_text})
    summarized = summarize_chain.invoke({"text": translated})
    
    # 输出结果
    print(f"输入文本: {input_text}")
    print(f"翻译结果: {translated}")
    print(f"总结结果: {summarized}")

# ==============================================
# 方法3: 顺序链 - 多输出接力
# 知识点: 选择性传递多个变量
# ==============================================
def demo_multi_output_chain():
    """演示顺序链 - 多输出接力"""
    print("\n" + "="*50)
    print("【方法3】多输出顺序链 (Multi-output Chain)")
    print("-"*40)
    
    # 步骤1: 分析文本（输出JSON）
    analyze_prompt = PromptTemplate.from_template(
        "分析以下文本，提取主题和情感。\n文本：{input_text}\n输出JSON格式：{{\"topic\": \"主题\", \"sentiment\": \"情感\"}}"
    )
    analyze_chain = analyze_prompt | llm | StrOutputParser()
    
    # 步骤2: 根据分析结果生成回复
    respond_prompt = PromptTemplate.from_template(
        "根据分析结果生成友好回复：\n主题: {topic}\n情感: {sentiment}"
    )
    respond_chain = respond_prompt | llm | StrOutputParser()
    
    # 执行
    analysis_result = analyze_chain.invoke({"input_text": "今天天气真好，我很开心！"})
    
    # 解析JSON并提取需要的变量
    import json
    analysis_dict = json.loads(analysis_result)
    
    # 只传递需要的变量
    response = respond_chain.invoke({
        "topic": analysis_dict["topic"],
        "sentiment": analysis_dict["sentiment"]
    })
    
    # 输出结果
    print(f"分析结果: {analysis_result}")
    print(f"提取的变量: topic={analysis_dict['topic']}, sentiment={analysis_dict['sentiment']}")
    print(f"最终回复: {response}")

# ==============================================
# 方法4: 数学计算链
# 知识点: 自然语言→Python代码→数值结果，避免大模型计算错误
# ==============================================
def demo_math_chain():
    """演示数学计算链"""
    print("\n" + "="*50)
    print("【方法4】数学计算链 (Math Chain)")
    print("-"*40)
    
    # 创建提示词模板
    math_prompt = PromptTemplate.from_template(
        "将以下数学问题转换为Python代码并计算结果：{question}\n只输出最终数值。"
    )
    
    # 创建链
    chain = math_prompt | llm | StrOutputParser()
    
    # 执行
    result = chain.invoke({"question": "237乘以891等于多少？"})
    
    # 验证结果
    actual = 237 * 891
    
    # 输出结果
    print(f"问题: 237乘以891等于多少？")
    print(f"LLM计算结果: {result}")
    print(f"Python计算结果: {actual}")
    print(f"提示: 大模型是token预测器，计算易出错，建议用Python执行")

# ==============================================
# 方法5: 多文档总结链
# 知识点: 将多个文档拼接成一个Prompt
# ==============================================
def demo_documents_chain():
    """演示多文档总结链"""
    print("\n" + "="*50)
    print("【方法5】多文档总结链 (Documents Chain)")
    print("-"*40)
    
    # 创建提示词模板（注意使用context变量）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是文档总结助手，根据提供的文档回答问题。"),
        ("human", "文档内容：\n{context}\n\n问题：{question}")
    ])
    
    # 准备文档
    docs = [
        Document(page_content="人工智能（AI）是计算机科学的一个分支，旨在创建智能机器。"),
        Document(page_content="机器学习是AI的核心技术，使计算机能够从数据中学习。"),
        Document(page_content="深度学习是机器学习的一种，使用多层神经网络。")
    ]
    
    # 文档格式化函数
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])
    
    # 创建链
    chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 执行
    result = chain.invoke({
        "docs": docs,
        "question": "什么是人工智能？"
    })
    
    # 输出结果
    print(f"文档数量: {len(docs)}")
    print(f"问题: 什么是人工智能？")
    print(f"回答:\n{result}")

# ==============================================
# 方法6: LCEL新式链式调用（推荐）
# 知识点: 使用|管道符，支持invoke/stream/ainvoke
# ==============================================
def demo_lcel():
    """演示LCEL新式链式调用"""
    print("\n" + "="*50)
    print("【方法6】LCEL链式调用 (推荐)")
    print("-"*40)
    
    # 创建提示词模板
    prompt = PromptTemplate.from_template("请用简洁语言解释{topic}。")
    
    # 创建链（最简洁的方式）
    chain = prompt | llm | StrOutputParser()
    
    # 执行invoke
    result1 = chain.invoke({"topic": "量子计算"})
    
    # 链式组合（添加自定义函数）
    def add_prefix(text):
        return f"【解释结果】\n{text}"
    
    chain_with_prefix = prompt | llm | StrOutputParser() | add_prefix
    result2 = chain_with_prefix.invoke({"topic": "区块链"})
    
    # 输出结果
    print(f"输入: 量子计算")
    print(f"输出:\n{result1}")
    print("\n" + "-"*40)
    print(f"输入: 区块链（带前缀）")
    print(f"输出:\n{result2}")
    print("\n支持的调用方式: invoke() / stream() / ainvoke()")

# ==============================================
# 主函数 - 演示所有方法
# ==============================================
def main():
    print("="*60)
    print("Chains 入门学习示例")
    print("="*60)
    
    # 逐个演示
    demo_basic_chain()
    demo_sequential_chain()
    demo_multi_output_chain()
    demo_math_chain()
    demo_documents_chain()
    demo_lcel()
    
    # 知识点总结
    print("\n" + "="*60)
    print("知识点总结")
    print("="*60)
    print("1. 基础链: Prompt | LLM | Parser 的组合")
    print("2. 顺序链: 多步骤接力，前一个输出作为下一个输入")
    print("3. 多输出传递: 可以选择性传递需要的变量")
    print("4. 数学计算链: 通过Python代码执行，避免大模型计算错误")
    print("5. 多文档总结链: 将多个文档拼接成Prompt")
    print("6. LCEL: 新式链式调用，使用|管道符，推荐使用")
    print("="*60)

if __name__ == "__main__":
    main()
