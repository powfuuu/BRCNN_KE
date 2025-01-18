import json

import json

def convert_bio_to_json(input_file_path, output_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        sentence = []
        ner = []
        current_entity = None  # 用于追踪当前实体
        for line in file:
            if line.strip():
                char, tag = line.strip().split()
                sentence.append(char)
                if tag.startswith('B-'):
                    # 如果有未结束的实体，先保存它
                    if current_entity:
                        ner.append(current_entity)
                    # 开始新的实体
                    current_entity = {"index": [len(sentence) - 1], "type": tag[2:]}
                elif tag.startswith('I-'):
                    # 继续当前实体
                    if current_entity and current_entity["type"] == tag[2:]:
                        current_entity["index"].append(len(sentence) - 1)
                    else:
                        # 如果当前实体类型不匹配，忽略不合法的 I- 标签
                        current_entity = None
                elif tag == 'O':
                    # 如果当前实体已完成，保存它
                    if current_entity:
                        ner.append(current_entity)
                        current_entity = None
            else:
                # 保存当前句子和实体
                if sentence:
                    if current_entity:
                        ner.append(current_entity)
                        current_entity = None
                    data.append({"sentence": sentence, "ner": ner})
                    sentence = []
                    ner = []
        # 处理文件末尾的句子
        if sentence:
            if current_entity:
                ner.append(current_entity)
            data.append({"sentence": sentence, "ner": ner})

    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, separators=(',', ':'))


# Define file paths
input_file_path = "D:\下载\cdtier\\test.txt"
output_file_path = "D:\PPSUC\workspace\BRCNN_KE\data\\cdtier\\test.json"

# Convert and save the file
convert_bio_to_json(input_file_path, output_file_path)
