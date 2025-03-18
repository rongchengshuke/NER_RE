import json
import random

# 定义生成随机坐标的函数
def generate_random_coordinates():
    return (random.randint(-90, 90), random.randint(-180, 180))


# 检查实体是否是地理实体相关
def is_geographical_entity(entity_names):
    keywords = ["地理实体", "地址", "地点", "地名"]
    return any(keyword in entity_names for keyword in keywords)


# 处理每一行的函数
def process_line(line):
    data = json.loads(line)  # 将每行json字符串转换为字典
    text = data["text"]
    entities = data["entities"]

    # 用一个集合来记录已经加过坐标的地理实体
    added_coordinates = set()

    for entity in entities:
        entity_names = entity.get("entity_names", [])

        if is_geographical_entity(entity_names):  # 如果是地理实体相关
            entity_text = entity["entity_text"]

            # 如果实体的entity_text中已经包含坐标，跳过处理
            if '(' not in entity_text and ')' not in entity_text:
                # 随机决定是否赋坐标，概率为0.6
                if random.random() < 0.6:
                    # 检查该实体是否已经处理过
                    if entity_text not in added_coordinates:
                        random_coordinates = generate_random_coordinates()
                        coordinates_str = f"({random_coordinates[0]},{random_coordinates[1]})"

                        # 更新实体文本和原始文本
                        entity["entity_text"] = f"{entity_text}{coordinates_str}"
                        text = text.replace(entity_text, f"{entity_text}{coordinates_str}")

                        # 将该实体加入已处理集合
                        added_coordinates.add(entity_text)

    # 返回处理后的结果
    data["text"] = text
    return json.dumps(data, ensure_ascii=False)


# 读取jsonl文件并处理
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            processed_line = process_line(line)
            outfile.write(processed_line + "\n")

# 示例调用
process_jsonl('./data/ccfbdci.jsonl', 'data/ccfbdci_NS.jsonl')
