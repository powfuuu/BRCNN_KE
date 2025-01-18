import json

def remove_pseudo_labeled_data(unlabeled_path, pseudo_labeled_path, output_path):
    # 读取未标注数据
    with open(unlabeled_path, 'r', encoding='utf-8') as f:
        unlabeled_data = json.load(f)

    # 读取伪标签数据
    with open(pseudo_labeled_path, 'r', encoding='utf-8') as f:
        pseudo_labeled_data = json.load(f)

    # 将伪标签数据的句子转换为集合
    pseudo_sentences = {tuple(item['sentence']) for item in pseudo_labeled_data}

    # 过滤掉未标注数据中出现在伪标签数据中的句子
    filtered_unlabeled_data = [item for item in unlabeled_data if tuple(item['sentence']) not in pseudo_sentences]

    # 保存过滤后的未标注数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_unlabeled_data, f, ensure_ascii=False, indent=None)


# 文件路径
unlabeled_path = '../data/cdtier/unlabeled.json'
pseudo_labeled_path = '../predict_cdtier_entropy_restrain-2.json'
output_path = '../data/cdtier/unlabeled.json'

# 文件路径
# unlabeled_path = '../predict_cdtier_0.98-5.json'
# pseudo_labeled_path = '../max_entropy-5.json'
# output_path = '../predict_cdtier_0.98-5.json'

# 执行过滤
remove_pseudo_labeled_data(unlabeled_path, pseudo_labeled_path, output_path)
