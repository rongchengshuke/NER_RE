from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
import csv
import random
import json
import re
import os

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

def convert_output_to_single_line(data):
    data["output"] = json.dumps(data["output"], ensure_ascii=False)  # 保持中文
    return data

def extract_and_parse_json(text):
    """
    从文本中提取 JSON 数据并进行解析。
    """
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        json_text = json_text.replace("\n", " ").replace("\t", " ").strip()

        try:
            data = json.loads(json_text)
            data = convert_output_to_single_line(data)
            return data
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
            return None
    else:
        print("未找到有效的 JSON 数据")
        return None

def save_json_list(json_list, filename):
    """
    将 JSON 列表写入文件，确保每个对象都有 instruction 字段。

    参数：
    json_list (list): 包含多个 JSON 对象的列表。
    filename (str): 要写入的 JSON 文件名。
    """

    person_org_relation = ["领导", "敌对", "合作"]
    org_org_relation = ["从属", "敌对", "合作"]
    equipment_org_relation = ["包含"]
    location_org_relation = ["驻扎", "占领", "防御", "支援"]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # 遍历 JSON 列表，确保每个对象都有 instruction 字段
    for item in json_list:
        if "instruction" not in item:
            item["instruction"] = "你是一个文本实体关系抽取领域的专家，你需要从给定的句子中提取出实体和关系. 以 json 格式输出, 如 {{\"entities\": [{{\"name\":\"外层抗击区临界线\",\"type\":\"军事装备\"}}],\"relations\": [{{\"subject\":\"北方联合军\",\"relation\":\"驻扎\",\"object\":\"兰州(36.06,103.79)\"}}]}}注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体和关系时, 输出\"没有找到任何实体和关系\". 3.如果地理实体有坐标需要输出地理实体的坐标，没有坐标则输出地理实体. 4.实体类型必须从一下四种实体类型进行选择：军事装备，地理位置，组织名称，人名。5.人员和组织之间的关系从{person_org_relation}中选择，组织和组织之间的关系从{org_org_relation}中选择，装备和组织之间的关系从{equipment_org_relation}中选择，地理位置和组织之间的关系从{location_org_relation}中选择".format(
                person_org_relation=person_org_relation,
                org_org_relation=org_org_relation,
                equipment_org_relation=equipment_org_relation,
                location_org_relation=location_org_relation
            )
    # 将数据写入 JSON 文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

def read_csv_first_column(csv_file):
    words = []
    with open(csv_file, 'r', encoding='GBK') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                words.append(row[0])
    return words

def convert_output_to_single_line(data):
    data["output"] = json.dumps(data["output"], ensure_ascii=False)  # 保持中文
    return data

def read_txt_to_list(filename):
    """
    读取 txt 文件的每一行，并存入列表。
    :param filename: 文件路径
    :return: 包含每一行内容的列表
    """
    lines = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # 去除空行和首尾空格
    return lines

if __name__ == '__main__':

    person_org_relation = ["领导", "敌对", "合作"]
    org_org_relation = ["从属", "敌对", "合作"]
    equipment_org_realation = ["包含"]
    location_org_relation = ["驻扎", "占领", "防御", "支援"]

    llm = CustomLLM_Siliconflow()
    template = '''
    【任务描述】
    根据所提供的军事类关键词生成一段1000-3000汉字字的战争类报道，文本中必须包含20-50个地理位置且对于一些地理位置给予坐标，
    文本中还要包含一些组织名称，人名。并且抽取出上面所提到的三种类型的实体，以及他们之间存在的明确的关系。

    【输入内容】
    关键词: {keywords}
    人员和组织之间的关系: {person_org_relation}
    组织和组织之间的关系: {org_org_relation}
    装备和组织之间的关系: {equipment_org_realation}
    地理位置和组织之间的关系: {location_org_relation}

    【要求】
    - 生成文本的语气要符合像记者记录时事一样，不要额外渲染文字，确保输出语句通顺，符合语法逻辑。
    - 所提供的关键词如果在{mili_list}列表中要抽取出军事装备实体类型，如果在{org_list}列表中要抽取出组织名称实体类型，不在这两个列表中可以适当舍弃，不作为实体被抽取出来。
    - 实体类型只有以下四种：军事装备，地理位置，组织名称，人名。关系类型根据所提供的关系字段进行选择。
    - 文本中生成的地理坐标的格式必须如下，南京(23,46)，其中(23,46)为南京的坐标；抽取的地理实体如果有坐标必须连同坐标一块抽取。
    - 输出为json格式，"input":"生成的本文","output":"抽取的实体和关系"。
    - output的标准输出格式为{{"entities": [{{"name": "南京(23,46)","type": "地理位置"}},{{...}}],"relations": [{{"subject": "315连","relation": "指挥下","object": "第5团"}},{{...}}]}}。
    '''

    csv_file = "军标库.csv"
    words_list = read_csv_first_column(csv_file)

    json_data_list = []  # 存储所有解析的 JSON 数据

    mili_list = read_txt_to_list("武器装备类.txt")
    org_list = read_txt_to_list("组织架构类.txt")

    index = 0
    for _ in range(8):  # 进行 5 次循环
        index += 1
        if index > 15:
            break
        random.shuffle(words_list)  # 重新打乱关键词列表
        num_keywords = random.randint(9, 13)
        selected_keywords = random.sample(words_list, min(num_keywords, len(words_list)))

        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(
            keywords=", ".join(selected_keywords),
            person_org_relation=person_org_relation,
            org_org_relation=org_org_relation,
            equipment_org_realation=equipment_org_realation,
            location_org_relation=location_org_relation,
            mili_list=mili_list,
            org_list=org_list
        )

        response_text = llm(prompt).lstrip("\n")
        print("原始响应:", response_text)  # 调试用，看看 API 返回的内容

        parsed_json = extract_and_parse_json(response_text)  # 解析 JSON
        if parsed_json:
            json_data_list.append(parsed_json)  # 只存入有效数据
        else:
            print("警告：某次生成的 JSON 解析失败，已跳过！")

    # 统一存入 JSON 文件
    save_json_list(json_data_list, "test_data/json_test_1.json")
    print(f"所有数据已保存到 test_data/json_test_1.json，总共 {len(json_data_list)} 条数据。")

