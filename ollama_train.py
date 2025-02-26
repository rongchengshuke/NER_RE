import json
import pandas as pd
import torch
from datasets import Dataset
from ollama import Client  # 修正：正确导入 Ollama 的 Client
from sklearn.metrics import f1_score, accuracy_score
import os


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型推理所需数据格式的新数据集
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
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{input_text}",
                "output": entity_sentence,
            }

            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def predict(messages, client):
    """
    使用 Ollama 进行推理
    """
    # 修正：明确传递参数
    response = client.chat(
        model="deepseek-r1:1.5b",  # 指定模型名称
        messages=messages  # 传入消息列表
    )
    return response['message']['content']  # 提取返回内容


def evaluate(predictions, true_labels):
    """
    计算 F1 和 Accuracy
    """
    # 假设 predictions 和 true_labels 都是包含实体的文本列表
    true_entities = [set(label.split(';')) for label in true_labels]
    pred_entities = [set(pred.split(';')) for pred in predictions]

    # 计算 F1 和 Accuracy
    all_f1 = []
    all_acc = []
    for true, pred in zip(true_entities, pred_entities):
        all_f1.append(f1_score(list(true), list(pred), average="micro"))
        all_acc.append(accuracy_score(list(true), list(pred)))

    avg_f1 = sum(all_f1) / len(all_f1)
    avg_acc = sum(all_acc) / len(all_acc)

    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    return avg_f1, avg_acc


# 初始化 Ollama 客户端
client = Client()  # 修正：初始化 Ollama 客户端

# 加载数据集
train_dataset_path = "./data/ccfbdci.jsonl"
train_jsonl_new_path = "./data/ccf_train.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

# 得到训练集
total_df = pd.read_json(train_jsonl_new_path, lines=True)
test_df = total_df.sample(n=20)

# 进行推理并评估
predictions = []
true_labels = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, client)  # 修正：传入 client 对象
    predictions.append(response)  # 保存预测结果
    true_labels.append(row['output'])  # 保存真实标签
    print(index, "------------------------------------")
    # print("真实值：" + actual_output)
    print("预测值：" + response)

# 计算 F1 和 Accuracy
evaluate(predictions, true_labels)