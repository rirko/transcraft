"""
手搓 Transformer · 练习题（第 1 批）
====================================
范围：张量基础、nn.Module 套路、Embedding、注意力（QKV）

做法：
  1. 把每道题里的 `...` 删掉，写成你的实现（题目下方有提示）
  2. 运行本文件：python practice_01.py
  3. 打 ✓ 的题通过；打 ✗ 的会提示哪里不对
  4. 全部 ✓ 后，再打开 answers_01.py 对比参考写法（做题时先别看！）

难度标注：★ 入门   ★★ 进阶   ★★★ 综合/选做
"""

import torch
import torch.nn as nn

torch.manual_seed(1234)


# ============================================================
# 题 1 ★ 造张量
# 让 make_ids 返回一个张量，内容是 [2, 0, 7, 1]
# 提示：torch.tensor(...)
# ============================================================
def make_ids():
    tensor_return = torch.tensor([2,0,7,1])
    return tensor_return
    ...


# ============================================================
# 题 2 ★ 形状与索引（3 小问）
# 已知二维张量：
#   x = [[10, 20, 30],
#        [40, 50, 60]]      形状 [2, 3]
# ------------------------------------------------------------
def get_shape(x):
    """返回 x 的形状。
    提示：x.shape 本身就很像 tuple，直接 return 它就行"""
    ...
    return x.shape


def get_element(x):
    """返回 x 里数值为 50 的那个元素。
    提示：两个下标定位，行和列都从 0 开始数"""
    return x[1,1]
    ...
    


def get_row(x):
    """返回 x 的第二行 [40, 50, 60]（整个一行，不是一个数）。
    提示：只用一个下标"""
    ...
    return x[1]

# ============================================================
# 题 3 ★★ 手写 softmax
# 注意力权重的核心公式：把一串打分变成"和为 1"的占比
#   softmax(s)[i] = exp(s[i]) / sum(exp(s))
# 要求：不许调用 torch.softmax，用 exp 和除法自己实现
# 提示：scores.exp() 可以对每个元素求 exp
# ============================================================
def my_softmax(scores):
    ...
    softmaxreturn = scores.exp()/scores.exp().sum()
    return softmaxreturn

# ============================================================
# 题 4 ★★ 凭记忆写出第 1 课的 ShankeModel
# 一个 nn.Module 子类：
#   __init__(self, vocab_size, embedding_dim)
#       1. super().__init__()
#       2. 挂一个 nn.Embedding 到 self 上（属性名随意）
#   forward(self, ids)
#       返回查表结果
# ============================================================
class ShankeModel(nn.Module):
    ...
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.moduleE = nn.Embedding(vocab_size, embedding_dim)

    def forward(self,ids):
        token_embedding = self.moduleE(ids)
        return token_embedding
    





# ============================================================
# 题 5 ★★ Embedding 的本质
# 实现 manual_lookup：效果和 embed(ids) 一模一样，
# 但【不许】写 embed(ids)，只许用下标操作。
# 提示：embed.weight 就是那张表（一个普通二维张量），查表 = 按行号取行
# ============================================================
def manual_lookup(embed, ids):
    ...
    return embed.weight[ids]

# ============================================================
# 题 6 ★★ 给 Q 打分
# q 形状 [4]（一个词的查询向量），K 形状 [3, 4]（三个词的键）
# 返回打分结果，形状 [3]：每一行 K[i] 与 q 的点积
# 提示：矩阵乘法 K @ q 刚好就是这个效果
# ============================================================
def dot_score(q, K):
    ...
    return K @ q

# ============================================================
# 题 7 ★★ 加权求和（对照注意力示意图）
# weights = [0.1, 0.2, 0.7]，V 形状 [3, 4]（三个词的值向量）
# 返回 0.1*V[0] + 0.2*V[1] + 0.7*V[2]，形状 [4]
# 要求：不许用 @，用乘法和加法一行行写，感受"加权"到底加了什么
# ============================================================
def weighted_sum(weight, V):
    ...
    return torch.tensor([weight[0]*V[0,0],weight[0]*V[0,1],weight[0]*V[0,2],weight[0]*V[0,3]]) + torch.tensor([weight[1]*V[1,0],weight[1]*V[1,1],weight[1]*V[1,2],weight[1]*V[1,3]]) + torch.tensor([weight[2]*V[2,0],weight[2]*V[2,1],weight[2]*V[2,2],weight[2]*V[2,3]])

# ============================================================
# 题 8 ★★★ 综合：补全自注意力 forward
# __init__ 已经写好（三套权重矩阵），请补全 forward：
#   1. Q = x 过 W_q     提示：self.W_q(x)
#   2. K = x 过 W_k
#   3. V = x 过 W_v
#   4. scores = Q 和 K 转置后的矩阵乘法，形状 [3, 3]
#      提示：K.transpose(-2, -1) 是转置
#   5. weights = scores 每行做 softmax（dim=-1）
#   6. out = weights 加权 V（这次可以用 @ 了）
#   最后 return out, weights（一次返回两个值，是个 tuple）
# ============================================================
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        ...
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores =  Q @ K.transpose(-2,-1)
        weights = torch.softmax(scores,dim = -1)
        out = weights @ V
        return (out,weights)


# ============================================================
# 题 9 ★★★ 选做：Embedding = one-hot × 矩阵
# 每个编号变成 one-hot 向量再乘权重矩阵，效果等于直接查表。
# 实现 onehot_lookup(ids, W)：W 是 [vocab_size, dim] 的权重矩阵
# 提示：F.one_hot(ids, num_classes=W.shape[0]).float() 得到 one-hot
#       然后让它 @ W
# ============================================================
def onehot_lookup(ids, W):
    ...


# ############################################################
# ################  以下自动判题区，不要改  ##################
# ############################################################

def check(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        return True
    except AssertionError as e:
        msg = str(e)
        print(f"  ✗ {name} —— 结果不符合预期" + (f"（{msg}）" if msg else ""))
    except Exception as e:
        print(f"  ✗ {name} —— {type(e).__name__}: {e}")
    return False


def test_1():
    ids = make_ids()
    assert torch.is_tensor(ids), "返回值必须是一个 torch 张量"
    assert torch.equal(ids, torch.tensor([2, 0, 7, 1])), "内容应为 [2, 0, 7, 1]"


def test_2():
    x = torch.tensor([[10., 20., 30.],
                      [40., 50., 60.]])
    assert tuple(get_shape(x)) == (2, 3), "形状应为 (2, 3)"
    assert float(get_element(x)) == 50.0, "应返回数值 50"
    assert torch.equal(get_row(x), torch.tensor([40., 50., 60.])), "应返回第二行"


def test_3():
    scores = torch.tensor([2.0, 1.0, 0.1])
    w = my_softmax(scores)
    assert torch.is_tensor(w), "返回值必须是张量"
    assert torch.allclose(w, torch.softmax(scores, dim=0), atol=1e-6), "结果和 torch.softmax 不一致"
    assert abs(w.sum().item() - 1.0) < 1e-6, "权重之和应为 1"


def test_4():
    m = ShankeModel(10, 4)
    assert isinstance(m, nn.Module), "必须是 nn.Module 的子类"
    out = m(torch.tensor([1, 2, 3]))
    assert out.shape == (3, 4), "输出形状应为 (3, 4)"


def test_5():
    embed = nn.Embedding(5, 3)
    ids = torch.tensor([4, 0, 2])
    assert torch.allclose(manual_lookup(embed, ids), embed(ids)), "结果应与 embed(ids) 一致"


def test_6():
    q = torch.tensor([1., 0., 2., 1.])
    K = torch.randn(3, 4)
    assert dot_score(q, K).shape == (3,), "输出形状应为 (3,)"
    assert torch.allclose(dot_score(q, K), K @ q, atol=1e-5), "结果应等于 K @ q"


def test_7():
    V = torch.randn(3, 4)
    w = torch.tensor([0.1, 0.2, 0.7])
    out = weighted_sum(w, V)
    assert out.shape == (4,), "输出形状应为 (4,)"
    assert torch.allclose(out, w @ V, atol=1e-5), "结果应等于加权求和"


def test_8():
    torch.manual_seed(7)
    m = SelfAttention(4)
    x = torch.randn(3, 4)
    result = m(x)
    assert result is not None, "forward 还没有 return"
    out, weights = result
    Q, K, V = m.W_q(x), m.W_k(x), m.W_v(x)
    expected_w = torch.softmax(Q @ K.transpose(-2, -1), dim=-1)
    assert torch.allclose(weights, expected_w, atol=1e-5), "weights 计算不对"
    assert torch.allclose(out, expected_w @ V, atol=1e-5), "out 计算不对"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5), "每行权重之和应为 1"


def test_9():
    W = torch.randn(6, 3)
    ids = torch.tensor([5, 1, 1, 3])
    assert torch.allclose(onehot_lookup(ids, W), W[ids], atol=1e-5), "结果应等于 W[ids]"


if __name__ == "__main__":
    tests = [
        ("题 1  造张量",          test_1),
        ("题 2  形状与索引",       test_2),
        ("题 3  手写 softmax",    test_3),
        ("题 4  凭记忆写模型",     test_4),
        ("题 5  Embedding 的本质", test_5),
        ("题 6  给 Q 打分",        test_6),
        ("题 7  加权求和",         test_7),
        ("题 8  补全自注意力",     test_8),
        ("题 9  one-hot 本质(选做)", test_9),
    ]
    print("=" * 52)
    print("练习检查结果（先做题，再运行本文件）")
    print("=" * 52)
    passed = sum(check(name, fn) for name, fn in tests)
    print("-" * 52)
    print(f"通过 {passed}/{len(tests)} 题")
    if passed == len(tests):
        print("全部通过！现在可以打开 answers_01.py 对比参考写法了。")
    else:
        print("去上面找打 ✗ 的题继续加油，改完再运行一次。")
