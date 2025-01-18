import json


def merge_predictions_with_train(train_path, predictions_path):
    # 读取原有的train_org.json文件
    with open(train_path, "r", encoding="utf-8") as train_file:
        train_data = json.load(train_file)

    # 读取生成的预测结果文件
    with open(predictions_path, "r", encoding="utf-8") as predictions_file:
        predictions_data = json.load(predictions_file)

    # 将预测结果中的数据加入到train_org.json的数据中
    train_data.extend(predictions_data)
    # for item in train_data:
    #     item['ner'] = []

    # 保存合并后的数据到train_org.json文件中
    with open(train_path, "w", encoding="utf-8") as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=None)


# 示例使用
train_path = "../data/cdtier/train.json"
predictions_path = "../predict_cdtier.json"



merge_predictions_with_train(train_path, predictions_path)
