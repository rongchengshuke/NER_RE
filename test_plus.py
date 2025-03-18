import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from sklearn.metrics import accuracy_score, f1_score
import logging
from datetime import datetime
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 生成带日期的日志文件名
log_filename = f"error_log_{datetime.now().strftime('%Y%m%d')}.txt"

# 设置日志
logging.basicConfig(
    filename=log_filename,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_chinese_numbers(text):
    # 只匹配双引号中的内容，且确保是连续的汉字或数字（支持括号内的数字）
    pattern = r'"([^"]*[\u4e00-\u9fa5\d][^"]*)"'
    matches = re.findall(pattern, text)
    return matches


def extract_entities(context):
    # 使用正则表达式匹配所有合法的{}对
    matches = re.findall(r'\{[^{}]*\}', context)

    # 输出匹配到的所有{}对以及数量
    print("匹配到的{}对数量:", len(matches))
    print("匹配到的{}对:", matches)

    extracted_data = []
    organization_list = []
    location_list = []
    whole_list = []
    name_list = []
    text_acc = ""
    organization_acc = ""
    location_acc = ""
    name_acc = ""

    # 累加字符串时，跳过包含None值的条目
    for i, match in enumerate(matches):
        print(f"第{i + 1}个{{}}对:", match)

        result = extract_chinese_numbers(match)
        print(f"提取的内容：", result)

        # 构建字典
        match_dict = {"entity_text": None, "entity_label": None}

        if len(result) >= 1:
            match_dict["entity_text"] = result[0]  # 确保赋值到字典的相应键
        if len(result) >= 2:
            match_dict["entity_label"] = result[1]

        extracted_data.append(match_dict)

        # 仅当 entity_text 和 entity_label 都不为 None 时，才累加
        if match_dict["entity_text"] is not None and match_dict["entity_label"] is not None:
            text_acc += match_dict["entity_text"] + "," + match_dict["entity_label"] + ";"
            if "组织" in match_dict["entity_label"]:
                organization_acc += match_dict["entity_text"] + ","
            if "地理" in match_dict["entity_label"] or "地点" in match_dict["entity_label"]:
                location_acc += match_dict["entity_text"] + ","
            if "人名" in match_dict["entity_label"]:
                name_acc += match_dict["entity_text"] + ","

        print("构建的字典:", match_dict)
        print("-----------------------------------------------")

    return extracted_data, text_acc, organization_acc, location_acc, name_acc

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input_text = data["text"]
            entities = data["entities"]
            match_names = ["地点", "人名", "地理实体", "组织"]

            entity_sentence = ""
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json["entity_text"]
                entity_names = entity_json["entity_names"]
                for name in entity_names:
                    if name in match_names:
                        entity_label = name
                        break

                entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""

            if entity_sentence == "":
                entity_sentence = "没有找到任何实体"

            message = {
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京(12,34)", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". 3.如果地理实体有坐标需要输出地理实体的坐标，如南京(12,34)，(12,34)为他的坐标，没有坐标则输出地理实体. """,
                "input": f"文本:{input_text}",
                "output": entity_sentence,
            }

            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """

    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京(12,34)", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". 3.如果地理实体有坐标需要输出地理实体的坐标，如南京(12,34)，(12,34)为他的坐标，没有坐标则输出地理实体. """

    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    try:
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        logging.error(f"Error processing input: {messages}\nException: {str(e)}")
        return None  # 遇到错误返回 None

def clean_DeepSeek_string(s):
    # 去掉从第一个字符到第一个</think>
    s = re.sub(r'^.*?</think>', '', s, flags=re.S)
    # 去掉空行
    s = re.sub(r'\s+', ' ', s)
    # 去掉```json和```
    s = re.sub(r'```json|```', '', s)
    return s.strip()


# 加载模型和tokenizer，
# model_dir = "/home/zsy/NER-RE/models/Qwen2-1.5B"
model_dir = "/home/zsy/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/train_2025-03-18-07-49-58"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype=torch.bfloat16)

# 加载、处理数据集和测试集
train_dataset_path = "/home/zsy/NER-RE/data/ccfbdci_NS.jsonl"
train_jsonl_new_path = "/home/zsy/NER-RE/data/ccf_train_NS.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

# 得到测试集
total_df = pd.read_json(train_jsonl_new_path, lines=True)
# test_df = total_df[:int(len(total_df) * 1)].sample(n=20)
test_df = total_df[:int(len(total_df) * 0.5)].sample(n=1000, random_state=42)

# 准备预测
predictions = []
organization_predictions = []
location_predictions = []
name_predictions = []
labels = []
organization_labels = []
location_labels = []
name_labels = []

start_time = datetime.now()
print(f"开始时间: {start_time}")
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    actual_output = row['output']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    predicted_output = predict(messages, model, tokenizer)
    predicted_output = clean_DeepSeek_string(predicted_output)
    predicted_output = "没有找到任何实体" if "没有找到任何实体" in predicted_output else predicted_output

    if predicted_output is None:
        continue  # 跳过错误数据

    print(index, "------------------------------------")
    if actual_output != "没有找到任何实体":
        extracted_data, text_acc, organization_acc, location_acc, name_acc = extract_entities(actual_output)
        labels.append(text_acc)
        organization_labels.append(organization_acc)
        location_labels.append(location_acc)
        name_labels.append(name_acc)

        print("真实值：" + text_acc)
        print("真实值组织：" + organization_acc)
        print("真实值位置：" + location_acc)
        print("真实值名称：" + name_acc)
    else:
        labels.append("没有找到任何实体")
        organization_labels.append("没有找到任何实体")
        location_labels.append("没有找到任何实体")
        name_labels.append("没有找到任何实体")

        print("真实值：" + "没有找到任何实体")
        print("真实值组织：" + "没有找到任何实体")
        print("真实值位置：" + "没有找到任何实体")
        print("真实值名称：" + "没有找到任何实体")

    if predicted_output != "没有找到任何实体":
        extracted_data, text_acc, organization_acc, location_acc, name_acc = extract_entities(predicted_output)
        predictions.append(text_acc)
        organization_predictions.append(organization_acc)
        location_predictions.append(location_acc)
        name_predictions.append(name_acc)

        print("预测值：" + text_acc)
        print("预测值组织：" + organization_acc)
        print("预测值位置：" + location_acc)
        print("预测值名称：" + name_acc)
    else:
        predictions.append("没有找到任何实体")
        organization_predictions.append("没有找到任何实体")
        location_predictions.append("没有找到任何实体")
        name_predictions.append("没有找到任何实体")

        print("预测值：" + "没有找到任何实体")
        print("预测值组织：" + "没有找到任何实体")
        print("预测值位置：" + "没有找到任何实体")
        print("预测值名称：" + "没有找到任何实体")
    print(index, "------------------------------------")

    # predictions.append(predicted_output)
    # labels.append(actual_output)

    # print("真实值：" + text_acc)
    # print("预测值：" + predicted_output)
    # print("真实值组织：" + organization_acc)
    # print("预测值组织：" + predicted_output)
    # print("真实值位置：" + location_acc)
    # print("预测值位置：" + predicted_output)
    # print("真实值名称：" + name_acc)
    # print("预测值名称：" + predicted_output)


end_time = datetime.now()
print(f"结束时间: {end_time}")

def score(labels, predictions):
    # 计算准确率和F1分数
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')  # 或使用 'micro', 'macro' 等
    return acc, f1

# 计算准确率和F1分数
acc, f1 = score(labels, predictions)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

organization_acc, organization_f1 = score(organization_labels, organization_predictions)
print(f"组织Accuracy: {organization_acc:.4f}")
print(f"组织F1 Score: {organization_f1:.4f}")

location_acc, location_f1 = score(location_labels, location_predictions)
print(f"位置Accuracy: {location_acc:.4f}")
print(f"位置F1 Score: {location_f1:.4f}")

name_acc, name_f1 = score(name_labels, name_predictions)
print(f"人名Accuracy: {name_acc:.4f}")
print(f"人名F1 Score: {name_f1:.4f}")

elapsed_time = end_time - start_time
print(f"总运行时间: {elapsed_time}")



