import json

# 读取train.json文件
with open('../data/cdtier/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)


print('train_data length:', len(train_data))

with open('../data/cdtier/unlabeled.json', 'r', encoding='utf-8') as f:
    unlabeled_data = json.load(f)


print('unlabeled_data length:', len(unlabeled_data))

with open('../predict_cdtier.json', 'r', encoding='utf-8') as f:
    predict2_data = json.load(f)


print('predict_data length:', len(predict2_data))


