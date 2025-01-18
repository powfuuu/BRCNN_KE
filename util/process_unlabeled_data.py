import json


def process_sentence(sentence):
    # 将句子中的每个字符分隔开
    char_list = list(sentence)
    return {"sentence": char_list, "ner": []}


def process_file(input_file, output_file):
    result = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(process_sentence(line))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    input_file = "D:\桌面\\360.txt"  # 输入的txt文件路径
    output_file = '../data/cdtier/unlabeled.json'  # 输出的json文件路径

    process_file(input_file, output_file)
