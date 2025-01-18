import json


# 读取JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 保存JSON文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=None)


# 删除ner字段的值
def remove_ner_values(data):
    for item in data:
        item['ner'] = []
    return data


# 主函数
def main(input_file, output_file):
    # 加载数据
    data = load_json(input_file)

    # 删除ner字段的值
    modified_data = remove_ner_values(data)

    # 保存到新的文件中
    save_json(modified_data, output_file)
    print(f'Modified data has been saved to {output_file}')


# 指定文件路径并运行主函数
input_file = 'data/cdtier/test.json'
output_file = 'data/cdtier/test_without_ner.json'
main(input_file, output_file)
