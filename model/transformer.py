import math
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from model.layer_transformer import GraphConvolution


class GCN(nn.Module):
    ''' 仿GAMENet并在最后加入线性层：https://github.com/sjy1203/GAMENet/blob/master/code/models.py '''
    def __init__(self, vocab_tgt, emb_dim, adj, device):
        super(GCN, self).__init__()
        self.vocab_tgt = vocab_tgt
        self.emb_dim = emb_dim

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        # self.adj = torch.LongTensor(adj).to(device)
        self.x = torch.eye(vocab_tgt).to(device)

        self.gcn1 = GraphConvolution(vocab_tgt, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        # 两层全连接神经网络，每一层都是线性的
        # print("GCN")
        # print(vocab_tgt * emb_dim)
        # print(emb_dim * 4)
        # print(emb_dim)
        # print()
        self.linear1 = nn.Linear(vocab_tgt * emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)

    def forward(self):
        # print("forward")
        # print(self.x.size())
        # print(self.adj.size())
        # print()
        #  ======================================================================
        node_embedding = self.gcn1(self.x, self.adj)
        # print("shape: ", node_embedding.size())
        node_embedding = F.relu(node_embedding)
        # print("shape: ", node_embedding.size())
        node_embedding = self.dropout(node_embedding)
        # print("shape: ", node_embedding.size())
        node_embedding = self.gcn2(node_embedding, self.adj)  # [132, 128]
        #  ======================================================================
        # print("GCN shape: ", node_embedding.size())
        x = node_embedding.view(1, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        # print("GCN Dense shape: ", x.size())  # [1, 128]
        return x

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class EncoderDecoder(nn.Module):
    """标准的Encoder-Decoder架构"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # 源序列embedding
        self.tgt_embed = tgt_embed  # 目标序列embedding
        self.generator = generator  # 生成目标单词的概率

    def forward(self, src, tgt, src_mask, tgt_mask):
        "接收和处理原序列,目标序列,以及他们的mask"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """定义标准的linear+softmax生成步骤"""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "产生N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Encoder部分

class Encoder(nn.Module):
    """N层堆叠的Encoder"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "每层layer依次通过输入序列与mask"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """构造一个layernorm模块"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Add+Norm"""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "add norm"
        result = x + self.dropout(sublayer(self.norm(x)))
        return result


class EncoderLayer(nn.Module):
    """Encoder分为两层Self-Attn和Feed Forward"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Self-Attn和Feed Forward"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        result = self.sublayer[1](x, self.feed_forward)
        return result


# Decoder部分
class Decoder(nn.Module):
    """带mask功能的通用Decoder结构"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "将decoder的三个Sublayer串联起来"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# Attention
def attention(query, key, value, mask=None, dropout=None):
    "计算Attention即点乘V"
    d_k = query.size(-1)
    # [B, h, L, L]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    result = torch.matmul(p_attn, value), p_attn
    return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "初始化时指定头数h和模型维度d_model"
        super(MultiHeadedAttention, self).__init__()
        # 二者是一定整除的
        assert d_model % h == 0
        # 按照文中的简化，我们让d_v与d_k相等
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        实现MultiHeadedAttention。(实现多头注意力模型)
           输入的q，k，v是形状 [batch, L, d_model]。
           输出的x 的形状同上。
        """
        # 1) 第一步是计算一下mask。
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 2) 第二步是将这一批次的数据进行变形 d_model => h x d_k，即[batch, L, d_model] ->[batch, h, L, d_model/h]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 3) 针对所有变量计算注意力attn 得到attn*v 与attn
        # qkv :[batch, h, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 4) 上一步的结果合并在一起还原成原始输入序列的形状（将attention计算结果串联在一起，其实对张量进行一次变形：）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 最后再过一个线性层
        result = self.linears[-1](x)
        return result


# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    "实现FFN函数"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        result = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return result


# 生成 word 的embeddings
# 使用预学习的Embedding将输入Token序列和输出Token序列转化为dmodel维向量。
class Embeddings(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, d_model, ddi_adj, device, phy_flag=False, ddi_flag=False, src_flag=False):
        super(Embeddings, self).__init__()
        if src_flag:
            self.lut = nn.Embedding(vocab_src, d_model)  # vocab表示字典的大小 [4024, 128]
        else:
            self.lut = nn.Embedding(vocab_tgt, d_model)  # vocab表示字典的大小 [283, 128]
        self.ddi_adj = ddi_adj
        self.d_model = d_model  # 表示embedding的维度
        self.linear = nn.Linear(2, d_model)  # 将输入的生理特征用一个全连接层转化为一个向量
        self.phy_flag = phy_flag
        self.ddi_flag = ddi_flag
        self.src_flag = src_flag
        self.ddi_gcn = GCN(vocab_tgt, d_model, self.ddi_adj, device)

    def forward(self, x):
        nbatches = x.size(0)
        if self.src_flag:
            if self.phy_flag:
                if self.ddi_flag:
                    # print("x: ", x)
                    x = x[:, 2:]
                    # print("x: ", x)
                    result_diag = self.lut(x) * math.sqrt(self.d_model)  # [1024, 12, 128]
                    x_phy = x[:, :2].float()
                    # result_phy = self.linear(x_phy).view(nbatches, -1, self.d_model)
                    result_phy = self.linear(x_phy)
                    result_phy = result_phy.view(nbatches, -1, self.d_model)  # [1024, 1, 128]
                    result_ddi = self.ddi_gcn()  # [1, 128]
                    result_ddi = result_ddi.repeat(nbatches,1,1)  # [1024, 1, 128]
                    # print("ddi size: ", result_ddi.size())
                    # 按维数1拼接（横着拼）
                    result = torch.cat((result_phy, result_diag), 1)  # [1024, 13, 128]
                    # print("phy + diag size: ", result.size())
                    result = torch.cat((result, result_ddi), 1)  # [1024, 14, 128]
                    # print("phy + diag + ddi size: ", result.size())
                else:
                    x = x[:, 2:]
                    result_diag = self.lut(x) * math.sqrt(self.d_model)
                    x_phy = x[:, :2].float()
                    result_phy = self.linear(x_phy).view(nbatches, -1, self.d_model)
                    result = torch.cat((result_phy, result_diag), 1)
                    # print("phy + diag size: ", result.size())
            else:
                # 没有生理信息时，x就是诊断信息
                if self.ddi_flag:
                    result_diag = self.lut(x) * math.sqrt(self.d_model)
                    result_ddi = self.ddi_gcn()
                    result_ddi = result_ddi.repeat(nbatches,1,1)
                    # print("ddi size: ", result_ddi.size())
                    result = torch.cat((result_diag, result_ddi), 1)
                    # print("diag + ddi size: ", result.size())
                else:
                    result_diag = self.lut(x) * math.sqrt(self.d_model)
                    result = result_diag
                    # print("diag size: ", result.size())
        else:
            # print("x: ", x)
            result = self.lut(x) * math.sqrt(self.d_model)  # [1024, 13]
        return result

# 创建 positional encoding 并加入到 word embedding 中
class PositionalEncoding(nn.Module):
    "实现PE功能"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=True)
        result = self.dropout(x)
        return result


def make_model(src_vocab, tgt_vocab, ddi_adj, device, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, phy_flag=False, ddi_flag=False):
    "根据输入的超参数构建一个模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(src_vocab, tgt_vocab, d_model, ddi_adj, device, phy_flag, ddi_flag=ddi_flag, src_flag=True), c(position)),
                           nn.Sequential(Embeddings(src_vocab, tgt_vocab, d_model, ddi_adj, device, phy_flag, ddi_flag=ddi_flag, src_flag=False), c(position)),
                           Generator(d_model, tgt_vocab))

    # 使用xavier初始化参数，这个很重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


