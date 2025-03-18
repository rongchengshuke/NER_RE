from openai import OpenAI
from langchain.prompts import ChatPromptTemplate

# API密钥
API_KEY = "sk-xxxx"

# 自定义硅基流动大模型类
class CustomLLM_Siliconflow:
    def __call__(self, prompt: str) -> str:
        # 初始化OpenAI客户端
        client = OpenAI(api_key=API_KEY, base_url="http://18.191.125.135:3000/v1/")

        # 发送请求到模型
        response = client.chat.completions.create(
            model='deepseek-r1',
            messages=[
                {'role': 'user',
                 'content': f"{prompt}"}  # 用户输入的提示
            ],
        )

        # 打印响应结构，以便调试
        # print("Response structure:", response)

        # 收集所有响应内容
        content = ""
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    chunk_content = choice.message.content
                    # print(chunk_content, end='')  # 可选：打印内容
                    content += chunk_content  # 将内容累加到总内容中
        else:
            raise ValueError("Unexpected response structure")

        return content  # 返回最终的响应内容


if __name__ == '__main__':
    # 创建自定义LLM实例
    llm = CustomLLM_Siliconflow()

    # 基础版
    # 定义国家名称
    context = "而由南斯拉夫(-83,93)反对党所发起的大规模非暴力抗议活动5号进入了第4天，大约有20万名的群众聚集在南斯拉夫(-83,93)首都贝尔格勒，要求南国总统米洛舍维奇承认总统选举失败。"
    keyword = "反坦克火箭"

    template = '''
    【任务描述】
    请根据用户输入的内容执行以下两种操作之一：
    1. **同义词替换**：找到句子中与关键词相似的词，并替换为更合适的表达，使句子自然通顺。
    2. **续写**：如果句子较短，根据关键词补充 1-2 句，使其完整流畅。

    【输入内容】
    原始句子: {context}
    关键词: {keyword}

    【要求】
    - 叙述的语气要符合像记者记录时事一样，不要额外渲染文字。
    - 确保输出语句通顺，符合语法逻辑。
    - 保留句子中的坐标信息。
    - 如果原句中没有相似词，则根据关键词进行续写。
    - 只返回修改后的完整句子，不要额外的解释。
    '''

    # 使用模板创建提示
    prompt_template = ChatPromptTemplate.from_template(template)
    messages = prompt_template.format_messages(context=context, keyword=keyword)

    # 获取模型响应
    response = llm(messages).lstrip("\n")
    print(response)  # 打印响应内容