from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

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

'''
    一.需要掌握俩种实例化方法：
        1. PromptTemplate.from_template：不用写input_variables，自动提取模板中的变量名
        2. PromptTemplate：手动创建提示词模板
    二.需要掌握俩种方法：
        1. format：格式化提示词模板，将变量替换为实际值
        2. invoke：执行提示词模板，返回模型回复
    三.需要掌握5个参数：
        1. template：提示词模板字符串，包含变量名（如 {name}、{weather} 等）
        2. input_variables：模板中使用的变量名列表，用于指定模板中需要替换的变量
        3. partial_variables：预填充变量，初始编辑时写入的变量后续可以不再填写，若后续填写了，则会覆盖已填写的变量。
        4. validate：验证模板是否符合要求，默认值为 True
        5. validate_error：验证模板时的错误处理方式，默认值为 "raise"
    四.format与invoke不同的调用API：
        1.format:需要将字符串转为PromptValue对象
        2.invoke:直接返回模型回复，无需转为PromptValue对象
'''

template_str = "你好，{name}！今天天气{weather}。"

# ========== 1. 两种创建方式 ==========
def first_prompt_knowledge():
    print("=======/一.两种创建方式/=========")
    print("1. 使用 PromptTemplate 构造函数（需手动指定 input_variables）")
    prompt1 = PromptTemplate(
        template=template_str,
        input_variables=["name", "weather"]
    )
    print(prompt1.format(name="小明", weather="晴朗"))

    print("\n2. 使用 from_template（自动提取变量）")
    prompt2 = PromptTemplate.from_template(template_str)
    print(prompt2.format(name="小红", weather="多云"))

# ========== 2. format vs invoke ==========
def second_prompt_knowledge():
    print("=======/二.invoke与format格式化/=========")
    print("3.format 返回字符串")
    prompt = PromptTemplate.from_template(template_str)
    print(type(prompt.format(name="小红", weather="多云")))
    print("4.invoke 返回 PromptValue 对象")
    print(type(prompt.invoke({"name": "小红", "weather": "多云"})))

# ========== 3. partial_variables 示例 ==========
def get_current_weather():
    return "晴天（动态）"

# ========== 4. partial_variables ==========
def third_prompt_knowledge():
    print("=======/三.partial_variables 预填充变量/=========")
    print("5. 预填充变量")
    prompt = PromptTemplate.from_template(
        template_str,
        partial_variables={
            "weather": get_current_weather,   # 可以传函数，调用时动态求值
        }
    )
    print(prompt.format(name="李华"))  # weather 会自动调用函数
    # 覆盖 partial_variables 的值
    print("\n6. 二次填充可覆盖 partial_variables 的值")
    print(prompt3.format(name="李华", weather="暴雨"))  # 这里 weather 被覆盖为 "暴雨"

def fourth_prompt_knowledge_invoke():
    print("=======/四.invoke调用/=========")
    print("7. invoke 调用模型")
    #编写提示词
    prompt = PromptTemplate.from_template(template_str)
    prompt_value = prompt.invoke({"name": "李华", "weather": "暴雨"})
    print("invoke 返回的类型：", type(prompt_value))
    # 调用模型
    result = chat_model.invoke(prompt_value)
    # 调用模型，返回模型回复
    print(result.content)

def fourth_prompt_knowledge_format():
    print("=======/四.format调用/=========")
    print("8. format 调用模型")
    prompt = PromptTemplate.from_template(template_str)
    prompt_value = prompt.format(name="李华", weather="暴雨")
    print("format 返回的类型：", type(prompt_value))
    message = HumanMessage(content=prompt_value)
    print("转换后message 类型：", type(message))
    # 调用模型
    result = chat_model.invoke(message)
    print(result.content)


print("\n请输入：")
print("""a. 两种创建方式
b. format vs invoke
c. partial_variables 示例
d. invoke 调用模型
e. format 调用模型
""")    
i=input()
if i=="a":
    first_prompt_knowledge()
elif i=="b":
    second_prompt_knowledge()
elif i=="c":
    third_prompt_knowledge()
elif i=="d":
    fourth_prompt_knowledge_invoke()
elif i=="e":
    fourth_prompt_knowledge_format()
else:
    print("输入错误")
