[paper](https://arxiv.org/pdf/2012.09164)
# 核心逻辑
Point Transformer的核心就是：**如何在不规则的散点上做Attention？**
# 输入处理（把“点”变成Token）
**核心思路：**

1. **Input 不是一张图，而是一个 List**：无序的点集。
    
2. **拆分处理**：一定要把 xyz（几何）和 rgb（语义）分开看。xyz是**索引**，用来建立连接。rgb是**内容**，用来计算特征。
        
3. **Embedding**：只对内容 (rgb) 做线性映射，把低维物理特征变成高维语义特征。
### Point Transformer 的输入

- **格式**：**点集（Set of Points）**。它没有行，没有列，只有一堆点。
    
- **无序性**：这一堆点在计算机里的存储顺序不重要。第1个点是“车轮”，第100个点是“车窗”，如果你把它们在数组里的顺序换一下，它依然是那辆车。
    
- **数据张量**：  
    假设我们有一个 Batch，里面有 B个样本，每个样本有 N个点。输入数据通常长这样：  
    $Input=(B,N,C_{in})$
    这里的 $C_{in}$​是输入通道数，通常包含 **坐标** 和 **原始特征**。
### 核心逻辑：双流机制 (Coordinates vs Features)

这是 Point Transformer 最关键的输入逻辑。在 ViT 里，位置编码加到特征上之后，坐标信息就“隐身”了。但在 Point Transformer 里，**坐标信息必须全程“活着”**。

我们需要把输入数据拆分为两个独立的部分：

#### A. 坐标流 (Pos / Coordinates) —— 骨架

- **符号**：p(position)
    
- **维度**：(B,N,3) ，即 (x,y,z)。
    
- **作用**：
    
    1. **找邻居 (k-NN)**：Transformer 需要知道谁和谁是邻居，这完全依赖坐标计算距离。
        
    2. **相对位置编码**：在计算 Attention 时，需要算出点 i和点 j的相对距离，这也依赖坐标。
        
- **特点**：坐标数据通常**不参与**矩阵乘法（MLP）的非线性变换，它保持原样，只用来做几何计算。
    

#### B. 特征流 (Features) —— 血肉

- **符号**：x(feature)
    
- **维度**：(B,N,C)。这个 C,可能是 3 (RGB颜色)，也可能是 1 (激光雷达反射强度)，或者更多（法向量等）。
    
- **作用**：这是模型真正要学习和处理的语义信息（比如“红色”、“平坦”、“圆柱形”）。
    
- **特点**：它会像 ViT 的 Token 一样，不断地通过 Linear 层、Attention 层，维度会变（从 3 变 32，变 64...）。
    

**总结**：在代码实现中，Point Transformer 的 Layer 输入通常不是一个张量，而是一个元组：(pos, x)。

 - pos 负责告诉模型“结构是什么样”。
 - x 负责告诉模型“内容是什么”。
 
###  初始嵌入 (Initial Embedding/Linear Projection)

在 ViT 中，有一个 Patch Embed 层，用卷积把 Patch 变成 Token。  
在 Point Transformer 中，我们需要把原始的特征（比如 RGB）映射到高维特征空间。

#### 步骤过程：

1. **准备输入**：  
    假设你有一个点云，有 1024 个点。每个点有坐标 (x,y,z)和颜色 (r,g,b)。
    p(坐标): (1024,3)
    x(原始特征): (1024,3)
    
        
2. **特征映射 (Linear Projection)**：  
    我们只对 x(特征) 进行操作。使用一个简单的 **MLP (多层感知机)** 或者 **1x1 卷积**。
    
    - **公式**：$x_{new}​=Linear(x_{old}​)$
     **维度变化**：从 (1024,3)映射到(1024,32)。
        
    - 现在，每个点都有了一个长度为 32 的特征向量，代表它的“语义嵌入”。
        
3. **坐标如何处理？**  
    这里有两种流派，Point Transformer (Zhao et al.) 采用的是比较纯粹的做法：
    
    - **坐标** **p** **不直接变身**。它原封不动地保留下来，作为后续计算 Attention 时“生成位置编码”的**原材料**。
(注：有些其他变体网络会把坐标 xyz 也作为一个特征拼接到 RGB 后面，变成 6 通道输入，但这不改变核心逻辑，坐标依然需要单独保留一份用于几何计算。)
# 构建“局部邻域”（代替ViT的Patch）
在ViT中，Self-Attention通常是在全局（或者大的Window）上做的。但在3D点云中，点太多（几万甚至几十万个），做全局Attention计算量会爆炸（$O(N^2)$）。

Point Transformer 采用了**局部注意力（Local Attention）机制。**

- **核心操作**：**k-NN (k-Nearest Neighbors)**
    
- **通俗解释**：  
    对于每一个中心点 i，并不是看全图所有的点，而是只在它周围找 k个最近的“邻居点” j。  
    这 k个邻居组成了这个中心点 i的“朋友圈”。  
    **Point Transformer 的 Attention 只在这个“朋友圈”内部发生。**
    
**对比 ViT**：这就像 CNN 里的卷积核滑动，或者 Swin Transformer 里的 Window。我们在点云的局部几何结构上做文章。
### Point Transformer Layer
这是最精彩的部分。它和标准 Transformer 有两个巨大的不同：**相对位置编码** 和 **向量注意力（Vector Attention）**。
![](assets/Point-transformer/pointtransformer2.png)
**公式：** 
$$ y_i = \sum\limits_{x_j \in X(i)} \rho(\gamma(\phi(x_i)-\psi(x_j)+\delta))\odot(\alpha(x_j)+\delta)$$
$$ \delta = \theta(p_i-p_j)$$
#### 1. 相对位置编码 (Relative Positional Encoding)
在 ViT 里，位置编码（PE）是加在输入上的。但在 Point Transformer 里，位置编码是**在 Layer 内部现算**的，而且它不仅加在特征上，还参与了注意力的计算。

1. **原料**：相对坐标 $p=p_i​−p_j$
    
2. **加工**：通过一个 **MLP**（通常是两层 Linear + ReLU）。
    
3. **产物 (** **$δ_{ij}$****)：**相对位置编码**。 这就好比把物理上的“距离 1 米，东北方”翻译成了神经网络能理解的“语义向量”。
        

**关键点**：这个 $δ_{ij}$​会被用两次！一次用来算权重（Attention Score），一次用来增强（Value）。
#### 2. 向量注意力 (Vector Attention)

这是 Point Transformer (Zhao et al.) 和标准 Transformer 最大的区别。

##### 1. 什么是标量注意力（Standard Transformer）？

- **ViT 的做法**：Query 和 Key 点积后，得到**一个数**（比如 0.8）。
    
- **含义**：我对这个邻居的**所有特征通道**的关注度都是 0.8。不管你是红色通道还是绿色通道，统统打 8 折。
    

##### 2. 什么是向量注意力（Point Transformer）？

- **PT 的做法**：通过 MLP 算出的权重是一个**向量**（比如 \[0.8, 0.2, 0.9, ...]，长度等于特征通道数）。
    
- **含义**：我对这个邻居的**不同特征通道**有不同的关注度！
    
    - “对于他的‘颜色特征’，我很重视 (0.8)。”
        
    - “对于他的‘纹理特征’，我不关心 (0.2)。”
        
    - “对于他的‘形状特征’，我非常重视 (0.9)。”
##### 3. 怎么算出来的？（通俗版公式）

它不使用点积（Dot Product），而是使用**减法（Subtraction）**和**MLP**。

- **Step A (计算关系)**：先把中心点 i的特征和邻居 j的特征做个变换（类似 Q 和 K），然后**相减**。或者直接把特征拼接。
    
- **Step B (注入位置)**：把上面算出的特征差值，加上**相对位置编码** **$δ_{ij}$**。
    
- **Step C (生成权重)**：把这一坨东西扔进一个 **MLP**，最后接一个 **Softmax**。
    
- **结果**：得到一个维度为 (C)的权重向量。

#### 3.特征聚合 (Aggregation)
有了“权重向量”，有了“原材料”，最后一步就是融合。

1. **准备 Value**：  
    取邻居 j的特征 $x_j$​做线性变换，然后**加上相对位置编码 $δ_{ij}$​
     注意：这里再次把位置信息加进去了！这意味着输出的特征里包含了“他在哪”的信息。
        
2. **加权 (Element-wise Product)**：  
    使用**逐元素相乘**（Hadamard Product, ⊙）。
    
    - Value向量 ×权重向量。
        
    - 比如 Value 是 \[10, 20]，权重是 \[0.5, 0.1]，结果就是 \[5, 2]。
        
3. **求和 (Summation)**：  
    把 k个邻居算出来的结果加在一起，得到中心点 i的新特征。
    
4. **残差连接 (Residual Connection)**：  
    别忘了把中心点 i原始的输入特征加回来（Add），这和 ViT 一样，为了防止梯度消失。


# 整体架构 —— U-Net 结构

![](assets/Point-transformer/pointtransformer1.png)
ViT 通常是一个直筒子结构（一直保持 Token 数量不变）。  
但 Point Transformer 处理 3D 分割任务时，通常采用 **U-Net** 结构（编码器-解码器）。

这里涉及两个关键操作，对应 CNN 里的 Pooling 和 Upsampling：

1. **下采样 (Transition Down)**：
    
    - **目的**：减少点数，扩大感受野（Receptive Field），增加特征维度。
        
    - **做法**：使用**最远点采样 (Farthest Point Sampling, FPS)** 选出代表点（类似 Max Pooling 选最大的），然后用 k-NN 把周围点的特征聚合过来。
        
2. **上采样 (Transition Up)**：
    
    - **目的**：恢复点数，为了做逐点的语义分割。
        
    - **做法**：**三线性插值 (Trilinear Interpolation)**。把特征传回给被删掉的点，并把 Encoder 里的高分辨率特征拼过来（Skip Connection）。
        

