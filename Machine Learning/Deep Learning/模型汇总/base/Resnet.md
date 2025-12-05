[paper](https://arxiv.org/pdf/1512.03385)
**核心功能：** 训练深层网络
![[resnet _1.png]]
# 介绍
**目标：** 拟合H(x)
**设计残差方程：** F(x)=H(x)-x *#x是前层网络的训练输出结果*
**方法：** 中间层去拟合残差F(x)，最后将输出结果与x线性相加(***建立shortcut connections通道***)得到拟合的H(x)
***特别地，因为是使用加法，使得反向传播训练时不会导致梯度消失***
**通俗比喻：老板改稿**
想象你写了一篇文章（Input X）给老板看。

- **没有残差连接：** 老板把你的文章撕了，根据记忆重写了一篇。这很危险，老板可能改着改着把重点丢了。
    
- **有残差连接：** 老板拿红笔在你的原稿上进行**修改和批注**（Function(X)）。最后的成品 = 原稿 + 批注。
    
这样模型只需要学习**“这就好比只要学一点点改进（Residual）”**，而不是每次都从头学习整个表示，训练难度大大降低，模型也能堆得更深。


## Deeper Bottleneck Architectures
![[resnet_2.png]]
**框架：** The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller input/output dimensions.

# 网络设计
```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径的第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 主路径的第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接
        # 如果输入输出维度不一致（通道数或特征图尺寸），需要使用1x1卷积进行投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        # 主路径的前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 将主路径输出与捷径连接输出相加
        out += self.shortcut(residual)
        out = self.relu(out)

        return out
```
***只要使用了残差进行线性加和，网络就会自动训练$F(x)$,最后加上原输入$x$,而不是去计算目标函数$H(x)$
