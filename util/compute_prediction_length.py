import json

# 读取train.json文件
with open('../data/cdtier/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)


print('train_data length:', len(train_data))

with open('../data/cdtier/unlabeled.json', 'r', encoding='utf-8') as f:
    unlabeled_data = json.load(f)


print('unlabeled_data length:', len(unlabeled_data))

with open('../predict_cdtier_entropy_restrain-2.json', 'r', encoding='utf-8') as f:
    predict2_data = json.load(f)


print('predict_data length:', len(predict2_data))

# with open('../predict_cdtier_0.95-2.json', 'r', encoding='utf-8') as f:
#     predict2_data = json.load(f)
#
#
# print('predict2_data length:', len(predict2_data))

# with open('../max_entropy-5.json', 'r', encoding='utf-8') as f:
#     max_entropy2_data = json.load(f)
#
#
# print('max_entropy-5 length:', len(max_entropy2_data))
