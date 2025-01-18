import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# 读取JSON文件
with open('data/cdtier/test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化用于存储每种实体类别长度的字典
entity_lengths = defaultdict(list)

# 遍历数据，提取实体长度
for item in data:
    for entity in item['ner']:
        entity_type = entity['type']
        entity_length = len(entity['index'])  # 计算实体的长度
        entity_lengths[entity_type].append(entity_length)

# 计算每个实体种类的平均长度和方差
entity_types = []
average_lengths = []
variance_lengths = []

for entity_type, lengths in entity_lengths.items():
    entity_types.append(entity_type)
    average_lengths.append(np.mean(lengths))
    variance_lengths.append(np.var(lengths))  # 计算方差

# 绘制误差棒图
plt.figure(figsize=(10, 6))
plt.errorbar(entity_types, average_lengths, yerr=variance_lengths, fmt='o', capsize=5, capthick=2, elinewidth=2)

# 图表标题和标签
plt.title('Average Entity Length with Variance')
plt.xlabel('Entity Types')
plt.ylabel('Average Length')
plt.xticks(rotation=45)

# 显示图表
plt.tight_layout()
plt.show()
