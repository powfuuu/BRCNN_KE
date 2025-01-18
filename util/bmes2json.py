import json

def convert_bmes_to_json(input_file_path, output_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        sentence = []
        ner = []
        for line in file:
            if line.strip():
                char, tag = line.strip().split()
                sentence.append(char)
                if tag.startswith('B-'):
                    ner.append({"index": [len(sentence) - 1], "type": tag[2:]})
                elif tag.startswith('M-') or tag.startswith('E-'):
                    if ner:  # 检查ner列表是否为空
                        ner[-1]["index"].append(len(sentence) - 1)
                elif tag.startswith('S-'):
                    ner.append({"index": [len(sentence) - 1], "type": tag[2:]})
            else:
                if sentence:
                    data.append({"sentence": sentence, "ner": ner})
                    sentence = []
                    ner = []
        if sentence:
            data.append({"sentence": sentence, "ner": ner})

    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, separators=(',', ':'))

# Define file paths
input_file_path = "D:\下载\cdtier\\test.char.bmes"
output_file_path = "D:\PPSUC\workspace\W2NER\data\\cdtier\\test.json"

# Convert and save the file
convert_bmes_to_json(input_file_path, output_file_path)
