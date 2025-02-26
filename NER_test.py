from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import configparser
import json

# 获取模型路径
local_model_path = "./models/Qwen2-1.5B"

# 加载本地模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# 定义实体关系抽取模板
template = '''
【任务描述】
请根据以下背景知识，提取其中的实体以及它们之间的关系，并以结构化的形式返回实体及关系。你需要根据上下文识别实体、关系，并将其整理为键值对的形式。

【背景知识】
{{315连在第5团的指挥下攻占1234高地（12，54），44高地和上甘岭占地}}

【回答要求】
- 请列出从上下文中抽取的所有实体及其对应的关系。
- 输出应当是一个结构化的 JSON 格式，包含实体、关系和相关的属性。
- 抽取地点实体时，如果有经纬度坐标连通经纬度一起输出

{question}
'''


# 定义模型推理函数
def local_model_inference(context, question):
    # 构建输入文本
    input_text = template.format(context=context, question=question)

    # Tokenize 输入文本
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # 使用模型进行推理
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)

    # 解码并返回结果
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# 处理响应并解析 JSON 格式的结果
def extract_entities_and_relations(response):
    try:
        # 假设模型返回的文本已经是结构化的 JSON 格式
        structured_response = json.loads(response)
        return structured_response
    except json.JSONDecodeError:
        print("返回的结果无法解析为 JSON 格式")
        return None


# 测试
if __name__ == "__main__":
    context = "OpenAI 是一家人工智能公司，专注于自然语言处理技术，提供 API 服务。公司由 Sam Altman 和 Greg Brockman 创办。"
    question = "从以上背景知识中提取出实体及其关系。"

    # 获取模型响应
    response = local_model_inference(context, question)

    # 输出模型的响应
    print("模型的响应:", response)

    # 解析实体和关系
    entities_and_relations = extract_entities_and_relations(response)
    if entities_and_relations:
        print("提取的实体及关系:", json.dumps(entities_and_relations, ensure_ascii=False, indent=4))
