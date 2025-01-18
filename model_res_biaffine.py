import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class RDCNN_2D(nn.Module):
    def __init__(self,filters=300, kernel_size=(3, 3), num_block=4, dilation=[1, 2, 5]):
        super(RDCNN_2D, self).__init__()
        self.layers = [{"dilation": i} for i in dilation]
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_block = num_block
        print('!!!!!!!!!!!!!!!!!!!!!!RDCNN_2D_filter-{}_kernel-{}_block-{}_dilation-{}!!!!!!!!!!!!!!!!!!!'.format(filters, kernel_size, num_block, dilation))

        class Residual_IDCNN_block_2D(nn.Module):
            def __init__(self, layers, filters, kernel_size):
                super(Residual_IDCNN_block_2D, self).__init__()
                self.net = nn.Sequential()
                for i in range(len(layers)):
                    dilation = layers[i]["dilation"]
                    single_block = nn.Conv2d(in_channels=filters,
                                             out_channels=filters,
                                             kernel_size=kernel_size,
                                             dilation=dilation,
                                             padding=(dilation * (kernel_size[0] - 1) // 2,
                                                      dilation * (kernel_size[1] - 1) // 2))
                    self.net.add_module("layer%d" % i, single_block)
                    self.net.add_module("relu", nn.ReLU())

            def forward(self, X):
                Y = self.net(X)
                return F.relu(Y + X)

        self.rdcnn = nn.Sequential()
        for i in range(num_block):
            self.rdcnn.add_module("block%i" % i, Residual_IDCNN_block_2D(self.layers, self.filters, self.kernel_size))

    def forward(self, embeddings):
        output = self.rdcnn(embeddings)
        return output


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=1, padding=1) for _ in range(1)])

        print('luelueluelue')

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)


        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)

        outputs = x.permute(0, 2, 3, 1).contiguous()
        return outputs



class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class res_biaffine_output(nn.Module):
    def __init__(self, config, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, num_res_blocks, dropout=0):
        super(res_biaffine_output, self).__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        conv_input_size = cls_num + config.dist_emb_size + config.type_emb_size
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.dropout = nn.Dropout(dropout)

        print('res_biaffine_output with {} residual blocks'.format(num_res_blocks))

        class ResidualBlock(nn.Module):
            def __init__(self, conv_input_size, channels, cls_num, dropout):
                super(ResidualBlock, self).__init__()
                # self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
                self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
                self.mlp_rel = MLP(n_in=channels, n_out=cls_num, dropout=dropout)
                self.dropout = nn.Dropout(dropout)

            def forward(self, o1, dis_emb, reg_emb, grid_mask2d):
                o = torch.cat([dis_emb, reg_emb, o1], dim=-1)
                # o = torch.cat([reg_emb, o1], dim=-1)
                # o = torch.cat([dis_emb, o1], dim=-1)
                # print('去除区域嵌入')
                o = torch.masked_fill(o, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
                o = self.convLayer(o)
                o = self.dropout(self.mlp_rel(o))
                return o1 + o

        self.res_blocks = nn.ModuleList([ResidualBlock(conv_input_size, channels, cls_num, dropout) for _ in range(num_res_blocks)])

    def Rotary_position_embedding(self, qw, kw):
        batch_size, seq_len, output_dim = qw.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        pos_emb = position_ids * indices
        pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
        pos_emb = pos_emb.repeat((batch_size, *([1] * len(pos_emb.shape))))
        pos_emb = torch.reshape(pos_emb, (batch_size, seq_len, output_dim))
        pos_emb = pos_emb.to(qw)

        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.cat([-qw[..., 1::2], qw[..., ::2]], -1)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.cat([-kw[..., 1::2], kw[..., ::2]], -1)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, x, y, dis_emb, reg_emb, grid_mask2d):
        ent_sub = self.dropout(self.mlp1(x))
        ent_obj = self.dropout(self.mlp2(y))
        ent_sub, ent_obj = self.Rotary_position_embedding(ent_sub, ent_obj)
        # print('不去除旋转位置编码')
        o1 = self.biaffine(ent_sub, ent_obj)

        for block in self.res_blocks:
            o1 = block(o1, dis_emb=dis_emb, reg_emb=reg_emb, grid_mask2d=grid_mask2d)
        return o1

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0

        # self.bert = AutoModel.from_pretrained("/tmp/pycharm_project_321/chinese_roberta_wwm_ext", cache_dir="./cache/", output_hidden_states=True)
        self.bert = AutoModel.from_pretrained("D:\PPSUC\workspace\pretrained_model\\chinese_roberta_wwm_ext", cache_dir="./cache/", output_hidden_states=True)

        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size

        # self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
        self.dropout = nn.Dropout(config.emb_dropout)
        # self.predictor = res_biaffine_output(config, config.label_num, config.lstm_hid_size, config.biaffine_size,
        #                              config.conv_hid_size * len(config.dilation), config.ffnn_hid_size, config.num_res_blocks,
        #                              config.out_dropout)
        self.predictor = res_biaffine_output(config, config.label_num, config.lstm_hid_size, config.biaffine_size,
                                             config.conv_hid_size, config.ffnn_hid_size,
                                             config.num_res_blocks,
                                             config.out_dropout)

        # self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        # cln = self.cln(word_reps.unsqueeze(2), word_reps)
        #
        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)
        #
        # conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        # conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        # conv_outputs = self.convLayer(conv_inputs)
        # conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        # outputs = self.predictor(word_reps, word_reps, conv_outputs)
        outputs = self.predictor(word_reps, word_reps, dis_emb, reg_emb, grid_mask2d)

        return outputs
