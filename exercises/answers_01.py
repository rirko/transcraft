"""
手搓 Transformer · 练习题参考答案（第 1 批）
============================================
⚠️ 先自己动手做 practice_01.py，卡住了再来看！
每题答案只给核心写法，变量名可以不同，思路对就行。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 题 1：torch.tensor 把 Python 列表变成张量
def make_ids():
    return torch.tensor([2, 0, 7, 1])


# 题 2：索引从 0 开始数
def get_shape(x):
    return x.shape          # torch.Size([2, 3])，本身就是 (2, 3)

def get_element(x):
    return x[1, 1]          # 第 1 行、第 1 列 → 50

def get_row(x):
    return x[1]             # 只给行号 → 取出整行


# 题 3：exp 归一化。减 max 是防溢出的稳定技巧，不改变结果
def my_softmax(scores):
    e = (scores - scores.max()).exp()
    return e / e.sum()


# 题 4：第 1 课原样重现
class ShankeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, ids):
        return self.embedding_matrix(ids)


# 题 5：Embedding 的本质 = 一张权重表，查表 = 按行号取行
def manual_lookup(embed, ids):
    return embed.weight[ids]


# 题 6：矩阵乘列向量 = 每行分别与它做点积
def dot_score(q, K):
    return K @ q


# 题 7：加权求和展开写，就是示意图里那一步
def weighted_sum(weights, V):
    return weights[0] * V[0] + weights[1] * V[1] + weights[2] * V[2]


# 题 8：自注意力五步曲
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        Q = self.W_q(x)                     # 1. 查询
        K = self.W_k(x)                     # 2. 键
        V = self.W_v(x)                     # 3. 值
        scores = Q @ K.transpose(-2, -1)    # 4. 两两点积打分 [3, 3]
        weights = torch.softmax(scores, dim=-1)  # 5. 每行变成占比
        out = weights @ V                   # 6. 加权求和
        return out, weights


# 题 9：one-hot 乘矩阵，等价于按 1 所在的位置取行 → 就是查表！
def onehot_lookup(ids, W):
    onehot = F.one_hot(ids, num_classes=W.shape[0]).float()  # [4, 6]
    return onehot @ W
