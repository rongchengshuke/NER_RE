import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re

def clean_json_string(s):
    """ 清理可能的无效字符，使其符合 JSON 格式 """
    # 去除可能的控制字符（如换行符、回车符等）
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    # 替换掉一些潜在的非法字符
    s = re.sub(r'\\x[0-9A-Fa-f]{2}', '', s)
    return s

def str2list(string):
    # 清理字符串
    cleaned_string = clean_json_string(string)
    # 使用正则表达式匹配所有 JSON 结构
    json_strings = re.findall(r'\{.*?\}', cleaned_string)
    # 将字符串转换为 Python 字典
    json_list = [json.loads(js) for js in json_strings]
    return json_list

def compute_prediction_score(actual_output, predicted_output):
    """ 计算预测得分：预测出的 entity_text 如果出现在真实值中，则加一分 """
    actual_entities = {entity["entity_text"] for entity in actual_output}

    # 添加检查和调试信息
    predicted_entities = set()
    for entity in predicted_output:
        # 如果存在 'text' 字段，则将其转换成 'entity_text'
        entity_text = entity.get("entity_text", entity.get("text"))
        if entity_text:
            predicted_entities.add(entity_text)
        else:
            print(f"警告: 预测输出缺少 'entity_text' 和 'text'，输出内容: {entity}")

    score = sum(1 for entity in predicted_entities if entity in actual_entities)
    return score

def predict(messages, model, tokenizer):
    """ 生成模型的预测结果 """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 加载模型和tokenizer
model_dir = "./models/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

# 读取数据集
train_jsonl_new_path = "./data/ccf_train.jsonl"
total_df = pd.read_json(train_jsonl_new_path, lines=True)
test_df = total_df[:int(len(total_df) * 1)].sample(n=20)

# 预测并计算得分
total_score = 0

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    actual_output = row['output']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    predicted_output = predict(messages, model, tokenizer)

    # 解析 JSON
    actual_output = str2list(actual_output)
    predicted_output = str2list(predicted_output)

    # 计算得分
    score = compute_prediction_score(actual_output, predicted_output)
    total_score += score

    print(f"样本 {index} 预测得分: {score}")
    print("真实值：", actual_output)
    print("预测值：", predicted_output)
    print("-" * 50)

print(f"总得分: {total_score}")
