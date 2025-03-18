from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
import csv
import json
import re

# API密钥
API_KEY = "sk-xxxx"


# 自定义硅基流动大模型类
class CustomLLM_Siliconflow:
    def __call__(self, prompt: str) -> str:
        client = OpenAI(api_key=API_KEY, base_url="http://18.191.125.135:3000/v1/")
        response = client.chat.completions.create(
            model='deepseek-r1',
            messages=[{'role': 'user', 'content': prompt}],
        )

        content = ""
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content += choice.message.content
        else:
            raise ValueError("Unexpected response structure")
        return content


# 读取 CSV 第一列
def read_csv_first_column(csv_file):
    words = []
    with open(csv_file, 'r', encoding='GBK') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                words.append(row[0])
    return words


# 提取 JSON 数据
def extract_and_parse_json(text):
    """
    从文本中提取 JSON 数据并进行解析。

    参数：
    text (str): 包含 JSON 数据的文本。

    返回：
    list 或 None: 如果成功解析 JSON，返回列表；否则返回 None。
    """
    json_match = re.search(r'\[.*\]', text, re.DOTALL)  # 改进正则表达式，匹配整个 JSON 数组
    if json_match:
        json_text = json_match.group(0)

        # 处理非法控制字符（去掉非 JSON 兼容字符）
        json_text = json_text.replace("\n", " ").replace("\t", " ").strip()

        try:
            # 解析 JSON
            data = json.loads(json_text)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
            return None
    else:
        print("未找到有效的 JSON 数据")
        return None


if __name__ == '__main__':
    llm = CustomLLM_Siliconflow()
    template = '''
    【任务描述】
    根据所提供的军事类关键词分成武器装备和组织架构两类，如果实在不符合这两类分到其他类别。

    【输入内容】
    关键词: {keywords}

    【要求】
    - 武器装备类包括军事武器，军事卫星，汽车，飞机等，军事配件
    - 组织架构类包括兵种，组织，中心，部队，所
    - 输出成 JSON 格式，如 [{{"type": "组织架构类", "name": "海军"}}]
    '''

    csv_file = "军标库.csv"
    words_list = read_csv_first_column(csv_file)

    org_list = []  # 组织架构类
    mili_list = []  # 武器装备类
    other_list = []  # 其他类别

    prompt_template = ChatPromptTemplate.from_template(template)
    for word in words_list:
        prompt = prompt_template.format(keywords=word)
        response = llm(prompt).lstrip("\n")
        print(response)

        json_data = extract_and_parse_json(response)
        if json_data:
            for item in json_data:
                if item.get("type") == "组织架构类":
                    org_list.append(item.get("name", ""))
                elif item.get("type") == "武器装备类":
                    mili_list.append(item.get("name", ""))
                else:
                    other_list.append(item.get("name", ""))

    # 写入文件
    with open("组织架构类.txt", "w", encoding="utf-8") as f_org:
        for org in org_list:
            f_org.write(org + "\n")

    with open("武器装备类.txt", "w", encoding="utf-8") as f_mili:
        for mili in mili_list:
            f_mili.write(mili + "\n")

    with open("其他类别.txt", "w", encoding="utf-8") as f_other:
        for other in other_list:
            f_other.write(other + "\n")

    print("分类完成！")
