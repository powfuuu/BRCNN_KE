import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import config
import data_loader
import utils
from model_res_biaffine import Model

import torch.nn.functional as F


def calculate_entropy(probas):
    """
    计算概率分布的熵。
    """
    return -torch.sum(probas * torch.log(probas + 1e-10), dim=-1)


def maximum_entropy_sampling(model, unlabeled_loader, config, n_instances, train_loader):
    """
    使用最大熵选择法来选择需要人工标注的样本。

    参数:
        model: 当前训练的模型
        unlabeled_loader: 无标签数据的 DataLoader
        config: 配置对象，包含批量大小等参数
        n_instances: 需要选择的样本数量

    返回:
        需要标注的样本
    """
    model.eval()
    entropies = []
    all_indices = []
    i = 0

    with torch.no_grad():
        for data_batch in unlabeled_loader:
            data_batch = [d.cuda() for d in data_batch[:-1]]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            length = sent_length

            grid_mask2d = grid_mask2d.clone()

            # 计算输出的置信度
            probs = torch.softmax(outputs, dim=-1)

            # 计算每个token对的熵
            entropy = calculate_entropy(probs)

            # 对每个样本的总熵求和
            sample_entropies = torch.sum(entropy, dim=(1, 2)).cpu().numpy()
            entropies.extend(sample_entropies)


    entropies = np.array(entropies)
    query_idx = np.argsort(entropies)[-n_instances:]


    return query_idx


def compute_entropy(probs):
    # 计算最后一个维度的熵值
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    # 在第一个和第二个 seq_len 维度上求平均
    return entropy.mean(dim=(0, 1)).cpu().numpy()


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = FocalLoss(alpha=0.25, gamma=2)

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)



    def predict(self, epoch, data_loader, train_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []
        entropy_results = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        total_entropy = 0
        total_samples = 0

        # 计算训练集的平均熵值
        with torch.no_grad():
            for data_batch in train_loader:
                # entity_text = data_batch[-1]
                data_batch = [d.cuda() for d in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

                # 计算输出的置信度
                probs = torch.softmax(outputs, dim=-1)
                # max_probs, preds = torch.max(probs, dim=-1)

                # 计算熵并累加
                for idx in range(probs.size(0)):
                    entropy = compute_entropy(probs[idx])
                    total_entropy += entropy
                    total_samples += 1

        # 计算训练数据集样本的平均熵值
        average_entropy = total_entropy / total_samples
        average_entropy = average_entropy / 2

        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [d.cuda() for d in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                # 计算输出的置信度
                probs = torch.softmax(outputs, dim=-1)
                max_probs, preds = torch.max(probs, dim=-1)

                # 对每个句子计算平均置信度
                sentence_confidences = max_probs.mean(dim=(1, 2)).cpu().numpy()

                # 计算这些样本的熵值
                for idx, conf in enumerate(sentence_confidences):
                    entropy = compute_entropy(probs[idx])
                    entropy_results.append((entropy, sentence_batch[idx], outputs[idx], entity_text[idx], length[idx]))

                # 预测实体
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())

                for idx, (ent_list, sentence) in enumerate(zip(decode_entities, sentence_batch)):
                    sentence_text = sentence["sentence"]
                    instance = {"sentence": list(sentence_text), "ner": []}
                    for ent in ent_list:
                        instance["ner"].append(
                            {"index": ent[0], "type": (config.vocab.id_to_label(ent[1])).capitalize()})
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        # 选择熵值低于平均熵值的样本
        # filtered_results = [item for item in entropy_results if item[0] < average_entropy]
        filtered_results = [item for item in entropy_results]


        filtered_list = []
        for entropy, sentence, output, entity_text, length in filtered_results:
            outputs = torch.argmax(output.unsqueeze(0), -1)
            ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), [entity_text],
                                                                [length.cpu().numpy()])

            for ent_list, sentence in zip(decode_entities, [sentence]):
                sentence_text = sentence["sentence"]
                instance = {"sentence": list(sentence_text), "ner": []}
                for ent in ent_list:
                    instance["ner"].append({"index": ent[0], "type": (config.vocab.id_to_label(ent[1])).capitalize()})
                filtered_list.append(instance)

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="micro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                             pred_result.numpy(),
                                                             average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        # with open(config.predict_path, "w", encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False, indent=None)

        # 保存低于平均熵选择的样本
        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(filtered_list, f, ensure_ascii=False, indent=None)

        return e_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cdtier.json')
    parser.add_argument('--save_path', type=str, default='./model_res_biaffine_cdtier.pt')
    # parser.add_argument('--save_path', type=str, default='./model_biaffine.pt')
    parser.add_argument('--predict_path', type=str, default='./predict_cdtier.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--num_res_blocks', type=int, default=4)

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader, unlabeled_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config)
    model.load_state_dict(torch.load(config.save_path))
    model = model.cuda()

    trainer = Trainer(model)

    # 使用训练好的模型对未标注的数据进行预测
    logger.info("Loading Unlabeled Data")# 需要实现load_unlabeled_data方法

    logger.info("Predicting Unlabeled Data")
    trainer.predict("Unlabeled", unlabeled_loader, train_loader, ori_data[-1])  # 需要实现trainer的predict方法，保存预测结果到文件中
    # trainer.predict("Unlabeled", unlabeled_loader, ori_data[-1])  # 需要实现trainer的predict方法，保存预测结果到文件中