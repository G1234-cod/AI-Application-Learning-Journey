# 导入必要的库
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
    XMLOutputParser,
    StrOutputParser
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 获取API密钥
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

# ==============================================
# 方法1: 字符串输出 (StrOutputParser)
# 知识点: StrOutputParser将输出转为字符串，适合直接展示
# ==============================================
def demo_str_output(topic):
    """演示字符串输出解析器"""
    print("\n" + "="*50)
    print("【方法1】字符串输出 (StrOutputParser)")
    print("-"*40)
    
    # 创建提示词模板
    prompt = PromptTemplate.from_template(
        "请简要介绍{topic}，用简洁的中文回答。"
    )
    
    # 创建链: 提示词 -> 模型 -> 解析器
    chain = prompt | chat_model | StrOutputParser()
    
    # 执行并获取结果
    result = chain.invoke({"topic": topic})
    
    # 输出结果
    print(f"提示词: {prompt.format(topic=topic)}")
    print(f"结果类型: {type(result)}")
    print(f"结果内容:\n{result}")

# ==============================================
# 方法2: CSV列表输出 (CommaSeparatedListOutputParser)
# 知识点: 模型输出以逗号分隔，解析为列表形式
# ==============================================
def demo_csv_output(topic):
    """演示CSV列表输出解析器"""
    print("\n" + "="*50)
    print("【方法2】CSV列表输出 (CommaSeparatedListOutputParser)")
    print("-"*40)
    
    # 创建提示词模板
    prompt = PromptTemplate.from_template(
        "请列出{topic}的5个主要领域，只返回领域名称，用逗号分隔。"
    )
    
    # 创建链
    chain = prompt | chat_model | CommaSeparatedListOutputParser()
    
    # 执行并获取结果
    result = chain.invoke({"topic": topic})
    
    # 输出结果
    print(f"提示词: {prompt.format(topic=topic)}")
    print(f"结果类型: {type(result)}")
    print(f"结果内容: {result}")
    print(f"列表长度: {len(result)}")

# ==============================================
# 方法3: JSON格式输出 (PydanticOutputParser)
# 知识点: 
# 1. 使用Pydantic模型定义数据结构
# 2. 通过get_format_instructions()获取格式说明
# 3. 返回的是Pydantic模型对象，不是JSON字符串
# 4. 可通过.dict()转为字典，.json()转为JSON字符串
# ==============================================
def demo_json_output(topic):
    """演示JSON格式输出解析器"""
    print("\n" + "="*50)
    print("【方法3】JSON格式输出 (PydanticOutputParser)")
    print("-"*40)
    
    # 步骤1: 定义数据结构（Pydantic模型）
    class ApplicationAreas(BaseModel):
        description: str = Field(description="简要描述")
        areas: list[str] = Field(description="应用领域列表")
        count: int = Field(description="领域数量")
    
    # 步骤2: 创建解析器
    parser = PydanticOutputParser(pydantic_object=ApplicationAreas)
    
    # 步骤3: 获取格式说明（替代手动文字说明）
    format_instructions = parser.get_format_instructions()
    
    # 步骤4: 创建提示词模板（包含格式说明）
    prompt = PromptTemplate.from_template(
        "请分析{topic}。\n{format_instructions}"
    )
    
    # 步骤5: 创建链并执行
    chain = prompt | chat_model | parser
    result = chain.invoke({
        "topic": topic,
        "format_instructions": format_instructions
    })
    
    # 输出结果
    print(f"提示词模板:\n{prompt.template}")
    print(f"结果类型: {type(result)}")
    print(f"提示: PydanticOutputParser返回的是模型对象，不是JSON字符串")
    print(f"描述: {result.description}")
    print(f"领域列表: {result.areas}")
    print(f"领域数量: {result.count}")
    
    # 知识点: 如何转为字典或JSON
    print("\n[转换示例]")
    print(f"转为字典: {result.dict()}")
    print(f"转为JSON字符串: {result.json()}")

# ==============================================
# 方法4: XML格式输出 (XMLOutputParser)
# 知识点:
# 1. 可直接在提示词中说明返回XML格式（无需解析器）
# 2. 也可使用XMLOutputParser.get_format_instructions()
# 3. 处理后的格式是字典，里面数据是XML格式
# ==============================================
def demo_xml_output(topic):
    """演示XML格式输出解析器"""
    print("\n" + "="*50)
    print("【方法4】XML格式输出 (XMLOutputParser)")
    print("-"*40)
    
    # 步骤1: 创建XML解析器，指定标签
    parser = XMLOutputParser(tags=["area", "name", "description"])
    
    # 步骤2: 获取格式说明（替代手动文字说明）
    format_instructions = parser.get_format_instructions()
    
    # 步骤3: 创建提示词模板
    prompt = PromptTemplate.from_template(
        "请列出{topic}的3个主要应用领域及简要描述。\n{format_instructions}"
    )
    
    # 步骤4: 创建链并执行
    chain = prompt | chat_model | parser
    result = chain.invoke({
        "topic": topic,
        "format_instructions": format_instructions
    })
    
    # 输出结果
    print(f"提示词模板:\n{prompt.template}")
    print(f"结果类型: {type(result)}")
    print(f"提示: XMLOutputParser返回的是字典，里面包含XML格式数据")
    print(f"结果内容:\n{result}")

# ==============================================
# 方法5: 直接获取原始响应 (.content)
# 知识点:
# 1. 对话模型使用.response.content直接获取字符串
# 2. 非对话模型本身就是字符串
# ==============================================
def demo_direct_output(topic):
    """演示直接获取模型响应"""
    print("\n" + "="*50)
    print("【方法5】直接获取原始响应 (.content)")
    print("-"*40)
    
    # 创建提示词模板
    prompt = PromptTemplate.from_template(
        "请用3句话介绍{topic}。"
    )
    
    # 直接调用模型（不使用解析器）
    response = chat_model.invoke(prompt.format(topic=topic))
    
    # 获取字符串内容（知识点: 对话模型使用.content获取）
    result = response.content
    
    # 输出结果
    print(f"提示词: {prompt.format(topic=topic)}")
    print(f"响应类型: {type(response)}")
    print(f"结果类型: {type(result)}")
    print(f"结果内容:\n{result}")

# ==============================================
# 主函数 - 演示所有方法
# ==============================================
def main():
    # 同一个主题
    topic = "人工智能的应用领域"
    
    print("="*60)
    print(f"当前演示主题: {topic}")
    print("="*60)
    
    # 逐个演示不同的输出方法
    demo_str_output(topic)
    demo_csv_output(topic)
    demo_json_output(topic)
    demo_xml_output(topic)
    demo_direct_output(topic)
    
    # 知识点总结
    print("\n" + "="*60)
    print("知识点总结")
    print("="*60)
    print("1. StrOutputParser: 输出字符串，适合直接展示")
    print("2. CommaSeparatedListOutputParser: 输出列表，逗号分隔")
    print("3. PydanticOutputParser: 输出Pydantic模型对象，需用.dict()/.json()转换")
    print("4. XMLOutputParser: 输出字典，包含XML格式数据")
    print("5. .content: 对话模型直接获取字符串内容")
    print("6. 链(Chain): 使用|操作符串联提示词、模型、解析器")
    print("="*60)

# ==============================================
# 运行主函数
# ==============================================
if __name__ == "__main__":
    main()

# ==============================================
# 使用说明
# ==============================================
# 1. 在当前目录创建 .env 文件
# 2. 文件内容: OPENAI_API_KEY=你的API密钥
# 3. 运行命令: python 8.返回值类型.py
# ==============================================
