[paper](https://arxiv.org/pdf/1706.03762)
# Embedding
## **词向量（Word Embedding）**
Transformer 会为“苹果”这个词，定义一串数字特征（假设维度是 512，也就是有 512 个属性）：

- **苹果 (ID 500) ->** [0.9, 0.1, 0.8, -0.2, ...]
    

虽然机器理解的和人类不一样，但我们可以这样想象）：

1. **第一位**：是不是食物？（0.9 是）
    
2. **第二位**：是不是动作？（0.1 不是）
    
3. **第三位**：是不是圆的？（0.8 是）
    
4. **第四位**：是不是有生命的？（-0.2 不是）  
    ...一直到第 512 位。
    
这样一来，“苹果”就不再只是一个代号“500”，而变成了一个**实实在在的、有含义的向量**。
## 嵌入矩阵
![[transformer2.png]]

**矩阵大小 = 词表大小 (行数) $\times$向量维度 (列数)**

- **行数（Rows）：** 字典的大小（比如 10,000 行，每一行对应一个词）。
    
- **列数（Columns）：** 向量的维度（比如 512 列，也就是每个词有 512 个特征）。
    

**这个巨大的词表，就是嵌入矩阵。**

**工作流程是这样的（查表）：**

1. 输入进来一个词 ID（比如“我”是 5 号）。
    
2. 模型跑到这个巨大的矩阵里，找到**第 5 行**。
    
3. 把这一整行数字（512 个数字）复制出来。
    
4. 这就得到了“我”的 Embeddings。
    
在 PyTorch 等代码库里，这层通常就叫 `nn.Embedding`，它本质上就是一个**查表操作（Lookup Table）**。
**训练过程：** 
- **初始状态：** 刚开始训练时，这个矩阵里的数字全是**随机生成的垃圾数**。
    
- **训练过程：** 随着 Transformer 阅读了互联网上几百亿字的文本，它在做填空题（比如预测下一个词）的过程中，会不断由错误的反馈来**自动调整**这些数字。
    
- **最终结果：** 训练结束后，这些数字就“学会”了代表语义。模型自己领悟了“苹果”和“梨”应该长得像，否则它就没法正确预测后面的句子。
## Positional Encoding（位置编码）
Transformer 有个“缺点”：它是一次性并行处理所有词的（不像以前的 AI 是按顺序读）。所以在它眼里，“我爱你”和“你爱我”是一样的，因为它不知道谁在前面，谁在后面。

**解决办法：位置编码。**

就像给每个进入考场的学生发一个**座位号**。

- 词向量本身包含“含义”。
    
- 位置编码包含“位置信息”。
    
- 把两者**相加**。
    

这样，模型既知道这个词是啥，也知道它在句子的第几个位置。
# Self Attetion机制
**核心思想是：** 当 AI 看到句子中的某个词时，它能**同时**查看句子里的其他所有词，并判断哪些词对理解当前这个词最重要。
**Self-Attention 就是要算出每个词和上下文的关系强度。**
## Q、K、V 向量

为了实现这个机制，Transformer 把每个词都拆解成了三个向量（也就是三串数字），分别叫 **Query (Q)**、**Key (K)**、**Value (V)**。

- **Query (Q - 查寻):** 就像是你手里拿着的**择偶标准**（比如：我要找个爱运动的）。
    
- **Key (K - 标签):** 就像是对方头上的**特征标签**（比如：我是爱运动的、我是宅男）。
    
- **Value (V - 内容):** 就像是对方**真实的内涵**（具体的长相、性格、灵魂）。
    
## 运作过程
**公式:** $Attention(Q, K, V ) = softmax(\frac{QK^T}{\sqrt d_k}  )V$
#### 步骤 1：MatMul（点积运算）—— 也就是“找匹配”

这是第一步，拿着你的 Q（查询意图）去和所有的 K（索引标签）做对比。

- 在数学上，两个向量做**点积（Dot Product）**，算出来的结果代表**相似度**。
    
- **结果：** 算出一个分数列表。
    
    - 苹果(Q) vs 手机(K) -> 分数 100（很高，很相关）
        
    - 苹果(Q) vs 香蕉(K) -> 分数 80（还行，都是水果）
        
    - 苹果(Q) vs 车轮(K) -> 分数 0（完全无关）
        

#### 步骤 2：Scale（缩放）—— 也就是“防止走极端”

这是论文标题中“Scaled”的由来，也是很多初学者容易忽略的一步。

- **动作：** 把上面算出来的分数，除以$\sqrt d_k$**（查询空间维度的平方根）**。
    
- **为什么要这么做？**
    
    - 如果向量维度很高，点积算出来的分数会非常大（比如几千、几万）。
        
    - 分数太大的话，下一步做 Softmax 的时候，就会出现**“胜者通吃”**的情况（最大的那个变成 1，其他全是 0）。
        
    - 这会导致梯度消失，AI 学不动了。所以要**除以一个数，把数值拉回到一个温和的范围内**。
        

#### 步骤 3：Mask（掩码，可选）—— 也就是“不准偷看”

**（这一步主要用在 Decoder 解码器里，Encoder 里通常不需要）。**

- 如果是生成任务，AI 不能看见未来的词。所以要把“未来”的位置的分数强行设为负无穷大（-∞），这样 AI 就会彻底忽略它们。
    

#### 步骤 4：Softmax（归一化）—— 也就是“算百分比”

把刚才经过缩放的分数，转化成**概率分布**（所有分数加起来等于 1）。

- 苹果 vs 手机 -> 100分 -> 变成 **60%**
    
- 苹果 vs 发布 -> 50分 -> 变成 **30%**
    
- 苹果 vs 其他 -> 低分 -> 变成 **10%**
    

#### 步骤 5：MatMul（加权求和）—— 也就是“提取内容”

最后一步，用算出来的百分比，去乘对应的 **V (Value)**。

- **$最终输出 = 0.6 \times(手机的V) + 0.3 \times(发布的V) + ...$**
    
- 如果不相关（百分比接近0），那个词的 V 就几乎不会被加进来。
    

**总结：** 这一套流程下来，“苹果”这个词向量吸取了它周围最重要词汇的信息，变成了一个**“融合了上下文信息的超级向量”**。

---



# Multi-Head Attention（多头注意力）
如果只有一个注意力模块（一个头），容易出现**偏科**
比如“苹果”这个词，如果这唯一的头只关注了它“作为水果”的属性，可能就忽略了它“作为科技公司”的属性。

**Multi-Head（多头）就是雇佣了一个“专家考察团”。**
**公式：**$MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O$
where $head_i = Attention(QW^Q_i , KW^K_i ,VW^V_i)$
#### 1. 怎么切分？（分头行动）

假设我们模型的向量维度是 **512 维**。我们要用 **8 个头 (Heads)**。  
Transformer **不是**搞了 8 个 512 维的矩阵（那样计算量会爆炸）。  
它是把 512 维**切开**：

- $512\div8 = 64$
    
- 每个头只负责处理 **64 维** 的数据。
    
- 大家并行计算，互不干扰。
    

#### 2. 每个头在看什么？（各司其职）

虽然数学公式是一样的，但因为参数初始化不同，训练后它们会学到不同的侧重点：

- **Head 1（语法专家）：** 专门看词性，关注“苹果”是主语还是宾语。
    
- **Head 2（指代专家）：** 专门看代词，如果后面有个“它”，Head 2 就会把注意力死死锁定在“苹果”上。
    
- **Head 3（语义专家）：** 关注“手机”、“发布”这些科技相关的词。
    
- **Head 4（距离专家）：** 专门关注离当前词最近的那个词。
    
- ...
    
#### 3. 拼接与融合 (Concat & Linear)

等这 8 个头都算完了，它们每人都拿着一个 64 维的结果向量。

- **Concat（拼接）：** 把 8 个 64 维的向量像排队一样拼回去 -> 变回 **512 维**。
    
- **Linear（线性层）：** 这是一次“高层会议”。也就是用一个大的权重矩阵$W^O$，把这拼起来的信息再做一次混合整理，统一思想，输出最终结果。
# Feed-Forward Networks & Residuals（前馈网络与残差）
注意力和位置搞定后，还需要经过一些标准的神经网络处理。
## Feed-Forward Networks (FFN)

在 Multi-Head Attention 结束之后，每个词的向量已经吸取了上下文的信息（比如“苹果”已经包含了“科技公司”的含义）。
但是，Attention 的操作全是**线性的**（加加乘乘），它的推理能力还不够强。**FFN 的作用，就是给模型增加“深度思考”的能力，并且把刚才看到的信息刻进脑子里。**

#### 1. 结构拆解：三明治结构

FFN 的结构非常简单，就是两个线性层（Linear）夹着一个激活函数（Activation）。

我们可以把它想象成一个**膨胀-收缩**的过程：

1. **第一层（扩充）：**  
    把输入的向量（比如 512 维）投影到一个**更高维的空间**（通常是 4 倍，即 2048 维）。
    
    - 为什么要变大？ 就像把一张折叠的纸展开，能看到更多的细节。在高维空间里，特征更容易被分离和理解。
        
2. **激活函数（ReLU/GELU）—— 这里是“非线性”的关键：**  
    给数据加点“弯曲”。如果没有这一步，再多的层堆起来也只是一个大的线性方程。激活函数让模型能理解复杂逻辑（比如：如果不...就...）。
    
    - GELU 是什么？ 现在的 LLM（大模型）多用 GELU，它比 ReLU 更平滑，就像一个光滑的滑梯，而不是生硬的折角。
        
3. **第二层（压缩）：**  
    把那个 2048 维的向量，再重新压缩回 512 维。
    
    - 目的是什么？ 提炼精华，去除冗余，保持和下一层接口一致。
        

#### 2. Position-wise（逐位置处理）

这一点非常重要！  
**在 Attention 层，词与词之间会疯狂交流（“苹果”看“手机”）。**  
**但在 FFN 层，大家谁也不理谁。**

每个词向量是**独立**进入 FFN 进行处理的。

- “我”进入 FFN -> 思考 -> 输出新的“我”。
    
- “爱”进入 FFN -> 思考 -> 输出新的“爱”。
    

**比喻：**  
Attention 是**小组讨论**，大家交换意见。  
FFN 是**独立自习**，每个人根据刚才讨论的结果，自己消化理解，内化成自己的知识。

#### 3. FFN 的神秘作用：键值记忆（Key-Value Memory）

最近的研究（如关于 GPT 的解释性研究）发现，**FFN 很可能存储了大量的事实性知识**。

- Attention 负责搞清楚“谁对谁”。
    
- FFN 负责记住“拿破仑是哪年死的”、“苹果公司的CEO是谁”。  
    那个中间膨胀的巨大维度（2048 或更多），就像是一个巨大的**神经元货架**，存储着模型学到的具体知识。
## Add (Residual Connection) —— 残差连接

在 Transformer 的每个子层（Attention 和 FFN）后面，都有一个 **Add** 操作。  
**Output = Input + Function(Input)**。
## Norm (Layer Normalization) —— 层归一化

在**Add**之后，紧接着就是 **Norm**。
#### 1. 为什么要做归一化？
在一个深层网络里，数据经过一层层的矩阵乘法，数值可能会波动很大。

- 有的变成了几千几万（爆炸）。
    
- 有的变成了 0.00001（消失）。  
    这会让模型很难训练，就像你在听歌，一会声音震耳欲聋，一会声音小得听不见。
    
#### 2. 怎么做？（强行拉平）
Layer Norm 会对每一个样本的向量进行统计：

1. 算出**平均值**（Mean）。
    
2. 算出**标准差**（Variance）。
    
3. **减去平均值，除以标准差。**
    
这就像是把班里所有同学的成绩，强行转换成**标准分**。不管这次卷子难还是简单，转换后大家的平均分都是 0，方差都是 1。

#### 3. 结果

数据变得**规矩**了。数值分布稳定在合理的范围内，梯度下降（训练过程）就会走得非常稳，收敛速度更快。
# 整体架构（Encoder 与 Decoder）
## 模型框架

![[transformer1.png]]
#### 1. Encoder（编码器）—— 负责“读”和“理解”

- 由一堆 Attention 和 FFN 堆叠而成。
    
- **任务：** 把输入的句子（比如中文“你好吗”）转化成一串包含深刻语义的向量矩阵。它把句子的所有特征、关系都揉碎了理解透了。
    

#### 2. Decoder（解码器）—— 负责“写”和“生成”

- 也是由堆叠层组成，但比 Encoder 多了一个机制：**Masked Attention（掩码注意力）**。
    
    - 为什么？ 生成句子时，你不能偷看后面还没生成的词。比如生成“I love you”，在生成“I”的时候，不能让模型看见“love”。所以要把后面的遮住（Mask）。
        

#### 3. Cross Attention（交叉注意力）—— 两者的桥梁

这是 Decoder 中最关键的一步。

- Decoder 在生成翻译结果时，会回头看 Encoder 的输出。
    
- **Q 来自 Decoder**（我现在生成到这一步了，我需要查什么信息？）。
    
- **K 和 V 来自 Encoder**（原文的理解结果）。
    
- 比喻： 就像翻译官，每写出一个英文单词，都要回头看一眼中文原文（Encoder 的输出），确认没翻错。

# 代码
## Encoder
### 输入部分 (Embeddings & Positional Encoding)
```python
import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 1. 词嵌入层：把 ID 查表变成向量
        # d_model: 向量维度 (比如 512)
        # vocab: 词表大小 (比如 10000)
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x 是输入的一串数字 ID
        # 这里的 math.sqrt(self.d_model) 是论文中的一个小细节
        # 作用是放大 embedding 的数值，为了和后面加上的 Positional Encoding 保持量级一致
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建一个矩阵 pe，用来存位置编码
        # max_len 是句子最大长度，d_model 是维度
        pe = torch.zeros(max_len, d_model)
        
        # 下面是论文中复杂的数学公式 (sin/cos)，你只需要知道它生成了一组固定的波形数字
        # 就像给每个位置发了一个独特的“纹身”
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置用 cos
        
        pe = pe.unsqueeze(0) # 增加一个 Batch 维度
        # register_buffer 告诉 PyTorch：这不是需要学习的参数，但这属于模型的一部分
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Sequence_Length, Embedding_Dim]
        # 把 Embedding 和 位置编码 直接相加
        x = x + self.pe[:, :x.size(1)]
        return x
```

### 核心组件 —— 多头注意力 (Multi-Head Attention)
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.h = heads  # 头数 (比如 8)
        self.d_k = d_model // heads # 每个头的维度 (512 // 8 = 64)
        
        # 定义四个线性层 (矩阵乘法)
        # W_q, W_k, W_v 负责把输入投影成 Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # W_o 负责最后把大家的结果融合
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # Batch Size (一次处理多少句话)
        
        # 1. 线性投影 (Linear Projections)
        # 此时形状: [Batch, Seq_Len, d_model]
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # 2. 切分多头 (Split Heads)
        # view: 把 d_model 拆成 (heads, d_k)
        # transpose: 把 heads 维度移到前面，方便并行计算
        # 变换后形状: [Batch, Heads, Seq_Len, d_k]
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # 3. 缩放点积注意力 (Scaled Dot-Product Attention)
        # Q 乘 K 的转置
        scores = torch.matmul(q, k.transpose(-2, -1)) 
        
        # 除以根号 d_k (缩放)
        scores = scores / math.sqrt(self.d_k)
        
        # 如果有 mask (掩码)，把不该看的地方设为负无穷
        if mask is not None:
             scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算概率分布
        attn = torch.softmax(scores, dim=-1)
        
        # 乘 V (提取信息)
        output = torch.matmul(attn, v)
        
        # 4. 合并多头 (Concat)
        # transpose: 把 heads 移回去
        # contiguous: 内存连续化(技术细节)
        # view: 把 (heads, d_k) 拼回 d_model
        # 形状变回: [Batch, Seq_Len, d_model]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        # 5. 最后的线性层 (Final Linear)
        output = self.out(output)
        
        return output
       ``` 

### 核心组件 —— 前馈网络 (FFN)
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        # d_ff 通常是 d_model 的 4 倍
        self.linear1 = nn.Linear(d_model, d_ff)   # 膨胀
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)   # 压缩

    def forward(self, x):
        # 也就是公式: ReLU(xW1 + b1)W2 + b2
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
```
### 组装层 (EncoderLayer)
``` python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        
        # 两个子层：Attention 和 FFN
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        
        # 两个归一化层 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # --- 子层 1: Attention ---
        # 原始输入 x 存起来做残差
        residual = x 
        
        # 跑 Attention
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        
        # Add & Norm: 原始 x + 新 x，再归一化
        x = self.norm1(x + residual)
        
        # --- 子层 2: FFN ---
        # 现在的 x 存起来做残差
        residual = x
        
        # 跑 FFN
        x = self.feed_forward(x)
        x = self.dropout(x)
        
        # Add & Norm
        x = self.norm2(x + residual)
        
        return x
```

**完整的编码器 (Encoder):** 最后，我们把多个 EncoderLayer 堆叠起来，就是整个 Encoder。
```python
import copy

# 一个小工具函数，用来克隆 N 层相同的网络
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super(Encoder, self).__init__()
        
        # 1. 嵌入层
        self.embed = Embeddings(d_model, vocab_size)
        
        # 2. 位置编码
        self.pe = PositionalEncoding(d_model)
        
        # 3. 堆叠 N 层 EncoderLayer (比如 6 层)
        self.layers = clones(EncoderLayer(d_model, heads), N)
        
        # 4. 最终的归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        # src 是输入的 token IDs
        
        # 先做 Embedding + 位置编码
        x = self.embed(src)
        x = self.pe(x)
        
        # 依次穿过 N 层 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)
            
        # 最后再做一次 Norm
        return self.norm(x)
```
### 测试代码
``` python
# 假设参数
vocab_size = 1000   # 词表大小
d_model = 512       # 向量维度
heads = 8           # 8个头
N = 6               # 堆叠6层

# 实例化模型
model = Encoder(vocab_size, d_model, N, heads)

# 模拟输入数据
# Batch size = 2, 句子长度 = 10
src = torch.randint(0, vocab_size, (2, 10)) 

# 这里的 mask 设为 None，假装没有 padding
output = model(src, None)

print("输入形状:", src.shape)   # [2, 10]
print("输出形状:", output.shape) # [2, 10, 512]
```
