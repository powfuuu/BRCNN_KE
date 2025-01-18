import json

# 加载 JSON 文件
with open("D:\PPSUC\workspace\BRCNN_KE\data\cdtier\\train_org_3543.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 删除长度超过500的句子
filtered_data = [item for item in data if len(item['sentence']) <= 500]

# 将修改后的内容保存回原文件
with open("D:\PPSUC\workspace\BRCNN_KE\data\cdtier\\train_org_3543.json", 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=None)

print(f"原始数据包含 {len(data)} 条句子，过滤后的数据包含 {len(filtered_data)} 条句子。")
print("过滤完成。")

# import json
#
# # 加载 JSON 文件
# with open('../data/cdtier/train_6602.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 找出长度大于500的句子
# long_sentences = [item['sentence'] for item in data if len(item['sentence']) > 500]
#
# # 打印长句子
# for i, sentence in enumerate(long_sentences, 1):
#     print(f"Sentence {i}: {''.join(sentence)}")
#     print(f"Length: {len(sentence)}")
