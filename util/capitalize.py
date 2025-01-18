import json


def capitalize_entity_types(file_path):
    # 读取生成的预测结果文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历数据，修改每个实体的type的首字母为大写
    for item in data:
        for entity in item.get("ner", []):
            entity["type"] = entity["type"].capitalize()

    # 将修改后的数据重新写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=None)


# 示例使用
# file_path = "../predict_cdtier_0.98-1.json"  # 假设生成的预测结果文件名为predictions.json
file_path = "../max_entropy-1.json"  # 假设生成的预测结果文件名为predictions.json
capitalize_entity_types(file_path)
