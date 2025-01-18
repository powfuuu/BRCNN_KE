import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
# from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
# import spacy
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# nlp = spacy.load("en_core_web_sm")

max_distance = 1000
dis2idx = np.zeros((max_distance), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text


    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def build_adj_matrix(dependencies, sentence_length, dep_type_to_id):
    adj_matrix = np.zeros((sentence_length, sentence_length), dtype=int)
    dep_type_matrix = np.zeros((sentence_length, sentence_length), dtype=int)

    for rel, gov, dep in dependencies:
        if gov >= 0 and dep >= 0:
            adj_matrix[gov][dep] = 1
            dep_type_matrix[gov][dep] = dep_type_to_id.get(rel, 0)  # 0 is default for unknown types

    return adj_matrix, dep_type_matrix




MAX_SEQ_LENGTH = 512

def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs[:MAX_SEQ_LENGTH-2] + [tokenizer.sep_token_id])

        length = min(len(instance['sentence']), MAX_SEQ_LENGTH - 2)
        _grid_labels = np.zeros((length, length), dtype=int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)
        _dist_inputs = np.zeros((length, length), dtype=int)
        _grid_mask2d = np.ones((length, length), dtype=bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens[:length]):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    distance = -_dist_inputs[i, j]
                else:
                    distance = _dist_inputs[i, j]
                if distance < max_distance:
                    _dist_inputs[i, j] = dis2idx[distance] + 9
                else:
                    _dist_inputs[i, j] = 19  # 如果距离超过范围，设置为默认值

        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index) or index[i] >= length or index[i + 1] >= length:
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            if index[-1] < length and index[0] < length:
                _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text

def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num



def load_data_bert(config):
    import json  # Don't forget to import json
    from transformers import AutoTokenizer  # Assuming you are using transformers for tokenization
    from collections import Counter  # If you are using a custom Vocabulary class
    import prettytable as pt  # If you are using PrettyTable for table representation

    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open('./data/{}/unlabeled.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        unlabeled_data = json.load(f)

    # tokenizer = AutoTokenizer.from_pretrained("/tmp/pycharm_project_321/chinese_roberta_wwm_ext", cache_dir="./cache/")
    tokenizer = AutoTokenizer.from_pretrained("D:\PPSUC\workspace\pretrained_model\\chinese_roberta_wwm_ext", cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    table.add_row(['unlabeled', len(unlabeled_data), 0])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    unlabeled_dataset = RelationDataset(*process_bert(unlabeled_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset, unlabeled_dataset), (train_data, dev_data, test_data, unlabeled_data)
    # return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)



