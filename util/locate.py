def find_all_word_positions(sentence, word):
    """
    计算一个词在句子中的所有位置，并打印每个匹配的起始和结束索引的所有索引。

    参数:
        sentence: 给定的中文句子 (字符串)
        word: 需要查找的词 (字符串)

    返回:
        包含所有匹配词位置的列表，每个位置是一个元组，包含该词在句子中的起始和结束位置（索引）。
    """
    positions = []
    start_index = 0

    while start_index < len(sentence):
        start_index = sentence.find(word, start_index)

        if start_index == -1:
            break

        end_index = start_index + len(word) - 1
        positions.append((start_index, end_index))

        # 打印从起始索引到结束索引之间的所有索引
        indices = list(range(start_index, end_index + 1))
        print(f"词 '{word}' 在句子中的索引范围: {indices}")

        # 更新 start_index，以查找下一个匹配位置
        start_index += len(word)

    return positions



# 示例
sentence = "攻击⽬标主要为巴基斯坦政府部⻔（如内阁部⻔）、巴基斯坦军⽅、军事⽬标等"
word = "军⽅"

position = find_all_word_positions(sentence, word)
# print(f"词 '{word}' 在句子中的位置: {position}")
