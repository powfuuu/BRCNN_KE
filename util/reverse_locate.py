import json

# 读取test.json文件
with open('../max_entropy-5.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理每个句子
for entry in data:
    sentence_list = entry['sentence']
    sentence = ''.join(sentence_list)
    print(f"Sentence: {sentence}")

    if 'ner' in entry and entry['ner']:
        for entity in entry['ner']:
            indices = entity['index']
            entity_text = ''.join([sentence_list[idx] for idx in indices])
            entity_type = entity['type']
            print(f"Entity: {entity_text}, Type: {entity_type}")
    print("\n")
