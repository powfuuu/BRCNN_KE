import json

# 假设json_data是包含上述JSON数据的字符串
with open('../max_entropy-2.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)


# 创建一个空集合用于存储不同的type值
# unique_types = set()

# # 遍历数据中的所有ner条目
# for item in train_data:
#     for entry in item.get('ner', []):
#         # 检查'index'键是否存在
#         if 'index' not in entry:
#             print(item)


# 打印所有不同的type值
# print(unique_types)

# for item in train_data:
#     for entry in item.get('ner', []):
#         # 检查'type'键是否存在
#         if 'type' not in entry:
#             print(item)


type_list = []
for item in train_data:
    for entry in item.get('ner', []):
        if entry['type'] not in type_list:
            type_list.append(entry['type'])
print(type_list)