# ==========================================
# LangChain 提示词模板参考
# 包含多种常用提示词模板模式
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
# 主要导入
from langchain_core.prompts import (
    PromptTemplate,          # 基础字符串模板
    ChatPromptTemplate,      # 对话格式模板
    MessagesPlaceholder,     # 动态消息占位符
    SystemMessagePromptTemplate,  # 系统消息模板
    HumanMessagePromptTemplate,   # 用户消息模板
    AIMessagePromptTemplate       # AI消息模板
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# 初始化配置
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

# 模板 2: 对话模板 ChatPromptTemplate
def template_chat():
    #对话模板 - 支持多轮对话历史,适用于需要上下文的对话场景
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}，请用{language}语言回答问题。"),
        ("human", "你好！"),
        ("ai", "你好！请问有什么可以帮助您的？"),
        ("human", "{user_input}")
    ])
    
    # 使用示例
    formatted_prompt = prompt.format_messages(
        role="工程师",
        language="中文",
        user_input="什么是 RESTful API？"
    )
    return formatted_prompt

# 模板 3: 带动态消息占位符
def template_with_history():
    #带历史消息占位符的模板,适用于需要动态插入对话历史的场景
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，根据对话历史回答用户问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])
    
    # 模拟对话历史
    chat_history = [
        HumanMessage(content="你好！"),
        AIMessage(content="你好！我是您的AI助手。")
    ]
    
    formatted_prompt = prompt.format_messages(
        chat_history=chat_history,
        user_input="你还记得我刚才说什么吗？"
    )
    return formatted_prompt

# ==========================================
# 模板 4: 结构化输出模板
# ==========================================
def template_structured_output():
    """
    结构化输出模板 - 指定输出格式
    适用于需要特定格式输出的场景
    """
    template = """
请按照以下JSON格式输出分析结果：
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "summary": "文本摘要",
    "keywords": ["关键词1", "关键词2"]
}}

待分析文本：{text}
"""
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )
    
    formatted_prompt = prompt.format(
        text="今天天气很好，我心情非常愉快！"
    )
    return formatted_prompt

# ==========================================
# 模板 5: 角色设定模板
# ==========================================
def template_role_play():
    """
    角色设定模板 - 让AI扮演特定角色
    适用于角色扮演、专家咨询等场景
    """
    system_prompt = """
    你现在是一位{profession}，拥有{years}年工作经验。
    请使用专业但易懂的语言回答问题，
    必要时可以提供具体的示例和建议。
    """
    
    human_prompt = "{question}"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])
    
    formatted_prompt = prompt.format_messages(
        profession="资深软件工程师",
        years=10,
        question="如何设计一个高并发系统？"
    )
    return formatted_prompt

# ==========================================
# 模板 6: 任务指令模板
# ==========================================
def template_task_instruction():
    """
    任务指令模板 - 明确任务步骤和要求
    适用于需要完成特定任务的场景
    """
    template = """
    请完成以下任务：
    
    任务：{task}
    
    要求：
    1. {requirement1}
    2. {requirement2}
    3. {requirement3}
    
    输入数据：{input_data}
    
    请输出详细的执行步骤和结果。
    """
    
    prompt = PromptTemplate(
        input_variables=["task", "requirement1", "requirement2", "requirement3", "input_data"],
        template=template
    )
    
    formatted_prompt = prompt.format(
        task="文本分类",
        requirement1="使用中文输出结果",
        requirement2="分类结果必须包含类别名称和置信度",
        requirement3="如果不确定可以返回'不确定'",
        input_data="这是一篇关于人工智能的新闻报道"
    )
    return formatted_prompt

# ==========================================
# 模板 7: 对比分析模板
# ==========================================
def template_comparison():
    """
    对比分析模板 - 比较多个对象
    适用于需要对比分析的场景
    """
    template = """
    请对比分析以下两个对象：
    
    对象A：{object_a}
    对象B：{object_b}
    
    请从以下维度进行对比：
    1. 优点
    2. 缺点
    3. 适用场景
    4. 性能指标
    
    输出格式：使用表格或分点列出。
    """
    
    #对比分析模板中的参数有object_a, object_b
    prompt = PromptTemplate(
        input_variables=["object_a", "object_b"],
        template=template
    )
    
    formatted_prompt = prompt.format(
        object_a="Python",
        object_b="Java"
    )
    return formatted_prompt

# ==========================================
# 模板 8: 逐步思考模板
# ==========================================
def template_step_by_step():
    """
    逐步思考模板 - 引导AI逐步推理
    适用于复杂问题求解场景
    """
    template = """
    请按照以下步骤解决问题：
    
    问题：{question}
    
    步骤1：理解问题，明确已知条件和目标
    步骤2：分析可能的解决方案
    步骤3：评估每个方案的优缺点
    步骤4：选择最优方案并说明理由
    步骤5：给出具体的实施步骤
    
    请详细描述每一步的思考过程。
    """
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    
    formatted_prompt = prompt.format(
        question="如何在3个月内学会Python编程？"
    )
    return formatted_prompt

# ==========================================
# 运行示例
# ==========================================
def main():
    # 1. 基础模板示例
    print("1. 基础模板:")
    print(template_basic())
    print("---\n")
    
    # 2. 对话模板示例
    print("2. 对话模板:")
    for msg in template_chat():
        print(f"{msg.type}: {msg.content}")
    print("---\n")
    
    # 3. 带历史消息模板示例
    print("3. 带历史消息模板:")
    for msg in template_with_history():
        print(f"{msg.type}: {msg.content}")
    print("---\n")
    
    # 4. 结构化输出模板示例
    print("4. 结构化输出模板:")
    print(template_structured_output())
    print("---\n")
    
    # 5. 角色设定模板示例
    print("5. 角色设定模板:")
    for msg in template_role_play():
        print(f"{msg.type}: {msg.content}")
    print("---\n")
    
    print("=" * 60)
    print("模板演示完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
