import json
import matplotlib.pyplot as plt
import numpy as np


# 计算每个数据集中实体类型的数量
def calculate_entity_type_counts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entity_type_counts = {}
    for item in data:
        for entity in item["ner"]:
            entity_type = entity["type"]
            if entity_type in entity_type_counts:
                entity_type_counts[entity_type] += 1
            else:
                entity_type_counts[entity_type] = 1
    return entity_type_counts


# 生成所有数据集中的所有标签，并自定义顺序
def get_all_entity_types(entity_type_counts, custom_order=None, sort_by_count=False):
    if custom_order:
        return custom_order  # 返回自定义顺序
    elif sort_by_count:
        # 按数量降序排列
        return sorted(entity_type_counts.keys(), key=lambda x: entity_type_counts[x], reverse=False)
    else:
        return sorted(entity_type_counts.keys())  # 默认按字母顺序排序


# 绘制玫瑰图（极坐标图），并在图中标注每种类型的数量和百分比
def plot_rose_chart(entity_type_counts, all_types, title):
    # 统一所有类型的顺序，即使某些类型在当前数据集中数量为0
    sizes = [entity_type_counts.get(entity_type, 0) for entity_type in all_types]
    total = sum(sizes)  # 计算实体总数

    # 计算每个扇形的角度
    angles = np.linspace(0, 2 * np.pi, len(all_types), endpoint=False).tolist()
    angles += angles[:1]
    sizes += sizes[:1]  # 闭合区域

    # 创建极坐标子图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 绘制玫瑰图
    ax.fill(angles, sizes, color='teal', alpha=0.3)
    ax.set_yticklabels([])  # 隐藏 y 轴标签
    ax.set_xticks(angles[:-1])  # 设置 x 轴标签
    ax.set_xticklabels(all_types)  # 设置实体类型为 x 轴标签

    # 添加每个类型的数量和百分比标签
    for i, (size, angle) in enumerate(zip(sizes[:-1], angles[:-1])):
        percentage = (size / total) * 100 if total > 0 else 0
        ax.text(angle, size + max(sizes) * 0.05, f'{size}\n({percentage:.1f}%)',
                horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

    # 显示图表
    # plt.title(title, fontsize=16)
    plt.show()


# 主函数
train_file = './data/cdtier/train.json'  # 替换为实际文件路径
test_file = './data/cdtier/test.json'  # 替换为实际文件路径
val_file = './data/cdtier/dev.json'  # 替换为实际文件路径

# 计算每个数据集的实体类型分布
train_counts = calculate_entity_type_counts(train_file)
test_counts = calculate_entity_type_counts(test_file)
val_counts = calculate_entity_type_counts(val_file)

# 自定义顺序或按数量排序
custom_order = ['Industry', 'Region', 'Tools', 'Campaign', 'Attacker']  # 自定义顺序
all_types = get_all_entity_types(train_counts, sort_by_count=False)  # 按数量排序

# 绘制各个数据集的玫瑰图
plot_rose_chart(train_counts, all_types, '训练集实体类型玫瑰图')
plot_rose_chart(test_counts, all_types, '测试集实体类型玫瑰图')
plot_rose_chart(val_counts, all_types, '验证集实体类型玫瑰图')
