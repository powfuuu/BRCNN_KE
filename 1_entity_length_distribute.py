import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 14})  # 设置字体大小

def calculate_entity_lengths(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entity_lengths = []

    for item in data:
        for entity in item["ner"]:
            length = len(entity["index"])  # 实体的长度为index列表的长度
            entity_lengths.append(length)

    return entity_lengths

def plot_histogram_with_density_dual_axis(entity_lengths):
    # 使用橙色和深绿色
    primary_color = '#2F4F4F'  # 橙色
    secondary_color = 'black'  # 深绿色

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制直方图（频率），使用柔和的颜色
    n, bins, patches = ax1.hist(entity_lengths, bins=20, alpha=0.7, color=primary_color, edgecolor='black')

    # 为每个柱子设置边缘颜色，保持整体简洁
    for patch in patches:
        patch.set_edgecolor('black')  # 设置边缘为黑色

    # 设置左边的Y轴标签为频率，增大字体
    ax1.set_xlabel('Entity Length', fontsize=18)  # 增大字体
    ax1.set_ylabel('Frequency', fontsize=18, color=primary_color)  # 增大字体
    ax1.tick_params(axis='y', labelcolor=primary_color)

    # 创建第二个Y轴，叠加密度曲线
    ax2 = ax1.twinx()

    sns.kdeplot(entity_lengths, ax=ax2, color=secondary_color, linewidth=2)  # 线条加粗

    # 使用Seaborn绘制密度曲线，颜色设置为深绿色
    sns.kdeplot(entity_lengths, ax=ax2, color=secondary_color)

    # 设置右边的Y轴标签为密度，增大字体
    ax2.set_ylabel('Density', fontsize=18, color=secondary_color)  # 增大字体
    ax2.tick_params(axis='y', labelcolor=secondary_color)

    # 设置图表背景为白色，简洁干净
    plt.gcf().set_facecolor('white')  # 设置整个图表的背景为白色

    # 显示图形
    plt.show()

def calculate_proportion(entity_lengths, threshold):
    total_count = len(entity_lengths)
    long_entities_count = sum(1 for length in entity_lengths if length > threshold)
    proportion = long_entities_count / total_count if total_count > 0 else 0
    return proportion

# 使用该函数读取文件并统计实体长度
file_path = './data/cdtier/test.json'  # 替换为你的文件路径
entity_lengths = calculate_entity_lengths(file_path)

# 绘制直方图和密度曲线（双纵坐标）
plot_histogram_with_density_dual_axis(entity_lengths)

# 计算长度超过10的实体所占的比重
threshold = 10
proportion = calculate_proportion(entity_lengths, threshold)
print(f"Proportion of entities longer than {threshold}: {proportion * 100:.2f}%")
