











"""手搓 Transformer · 第 1 课：Embedding 与注意力（QKV）
对应两张图：
  图1 ShankeModel —— 把 token ids 变成 token embeddings
  图2 注意力机制 —— 每个向量派生出 Q/K/V，加权求和得到新向量
运行方式：python 01_embedding_and_attention.py
"""
'''
import torch
import torch.nn as nn

torch.manual_seed(1234)                  # 固定随机种子，每次运行结果一致
torch.set_printoptions(precision=3, sci_mode=False)  # 打印时只显示3位小数，方便阅读


# ============================================================
# 第一部分：图1 的代码 —— 用 Embedding 把编号变成向量
# ============================================================

# 定义一个简单的模型, 用于将输入的 Token ids 转换为 token embeddings
class ShankeModel(nn.Module):

    # 初始化模型时调用
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()               # 固定仪式：先让父类 nn.Module 完成自己的初始化
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_dim)
        # ^ 一张 vocab_size 行 x embedding_dim 列的表格，里面全是随机小数（可训练参数）

    # 前向传播：数据流过模型的路径
    def forward(self, ids):
        token_embeddings = self.embedding_matrix(ids)  # 查表：取第 ids[0]、ids[1]... 行
        return token_embeddings


# 定义词表大小和词嵌入维度
vocab_size = 50275
embedding_dim = 768
model = ShankeModel(vocab_size, embedding_dim)   # 创建模型实例（自动执行 __init__）

# 随便指定一些 Token ids
ids = torch.tensor([1, 3, 5])

# 调用模型（自动触发 forward）
token_embeddings = model(ids)
print("=" * 60)
print("第一部分：Embedding 查表")
print("输入 ids 形状:", tuple(ids.shape))
print("输出的 token_embeddings 形状:", tuple(token_embeddings.shape))
print("其中第 1 个 token 的向量的前 5 个数:", token_embeddings[0, :5])
print()
'''

'''
# ============================================================
# 第二部分：图2 的代码 —— 注意力机制（QKV）
# ============================================================
# 为了能在屏幕上看清数字，这里用小尺寸演示：
# 3 个词「今天 / 天气 / 真」，每个向量只有 4 个数（对应图中的 4 个方块）

class SelfAttention(nn.Module):
    """自注意力层：每个 token 的向量 -> Q/K/V -> 加权求和"""

    # 初始化模型时调用
    def __init__(self, dim):
        super().__init__()
        # 三套可学习的权重矩阵。Q/K/V 不是复制，而是乘不同矩阵得到的三种"变形"
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    # 前向传播
    def forward(self, x):                 # x: 最下面那排向量，形状 [3, 4]
        Q = self.W_q(x)                   # 查询：我在找什么        [3, 4]
        K = self.W_k(x)                   # 键：我能被怎么匹配      [3, 4]
        V = self.W_v(x)                   # 值：我实际提供的信息    [3, 4]

        scores = Q @ K.transpose(-2, -1)  # 每个Q和每个K两两点积打分 -> [3, 3]
        # 注：真实实现这里还要除以 sqrt(dim) 防止打分过大，先省略以贴近示意图
        weights = torch.softmax(scores, dim=-1)  # 每行打分变成和为1的权重 -> [3, 3]
        out = weights @ V                 # 用权重对三个 V 加权求和 -> [3, 4]
        return out, weights


words = ["今天", "天气", "真"]            # 3 个 token
tiny_ids = torch.tensor([0, 1, 2])       # 它们各自的编号
embed = nn.Embedding(3, 4)               # 演示用小词表：3 个词，每个向量 4 个数
x = embed(tiny_ids)                      # 就是注意力图中最下面那排向量 [3, 4]

attn = SelfAttention(4)
out, weights = attn(x)

print("=" * 60)
print("第二部分：注意力机制（QKV）")
print("最底层的向量 x 形状:", tuple(x.shape))
print()
print("注意力权重矩阵 weights（3行x3列）:")
print(weights)
print()
print("怎么读这张表：第 i 行 = 第 i 个词对三个词的关注程度（每行加起来=1）")
for i, w in enumerate(words):
    dist = {words[j]: round(weights[i, j].item(), 2) for j in range(3)}
    print(f"  「{w}」的关注分布: {dist}")
print()
print("每个词吸收上下文后的新向量 out 形状:", tuple(out.shape))
print("「真」的新向量（图中写作 真'）:", out[2])
print()
print("=" * 60)
print("结论：Embedding 负责'把词变成向量'，注意力负责'让向量之间交流信息'。")
print("后面的课程就是不断打磨'交流信息'这个环节，最终堆出完整的 Transformer。")

'''