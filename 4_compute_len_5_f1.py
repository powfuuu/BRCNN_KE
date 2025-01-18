import json
from sklearn.metrics import precision_score, recall_score, f1_score


# 读取JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 过滤出长度超过5的实体
def filter_long_entities(entities):
    return [entity for entity in entities if entity['index'][-1] - entity['index'][0] + 1 > 10]


# 主函数
def main(test_file, output_file):
    # 加载数据
    test_data = load_json(test_file)
    output_data = load_json(output_file)

    true_labels = []
    pred_labels = []

    # 遍历每个样本
    for test_sample, output_sample in zip(test_data, output_data):
        # 过滤长度超过5的实体
        true_entities = filter_long_entities(test_sample['ner'])
        pred_entities = filter_long_entities(output_sample['ner'])

        # 提取实体位置和类型
        true_labels.extend([(tuple(entity['index']), entity['type']) for entity in true_entities])
        pred_labels.extend([(tuple(entity['index']), entity['type']) for entity in pred_entities])

    # 获取所有实体集合
    all_labels = sorted(set(true_labels + pred_labels))

    # 构建二元标签，用于计算微平均
    true_binary = [1 if label in true_labels else 0 for label in all_labels]
    pred_binary = [1 if label in pred_labels else 0 for label in all_labels]

    # 计算微平均的P值、R值和F1值
    p = precision_score(true_binary, pred_binary)
    r = recall_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary)

    # 输出结果
    print(f'Micro-averaged Precision: {p:.4f}')
    print(f'Micro-averaged Recall: {r:.4f}')
    print(f'Micro-averaged F1-Score: {f1:.4f}')


# 指定文件路径并运行主函数


print('BRCNN:')
test_file = 'data/cdtier/test.json'
output_file = './output_res_biaffine.json'
main(test_file, output_file)

print('BRCNN+KE:')
test_file = 'data/cdtier/test.json'
output_file = './output_res_biaffine_ke.json'
main(test_file, output_file)

print('w2ner:')
test_file = 'data/cdtier/test.json'
output_file = './output_w2ner.json'
main(test_file, output_file)

print('biaffine-ner:')
test_file = 'data/cdtier/test.json'
output_file = './output_biaffine.json'
main(test_file, output_file)
