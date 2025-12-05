[paper](https://arxiv.org/pdf/1612.02649)
# 作用
***与目标检测不同，语义分割可以识别并理解图像中每一个像素的内容：其语义区域的标注和预测是像素级的。***

# 转置卷积
***卷积不会增大输入的高宽，通常要么不变、要么减半；转置卷积则可以用来增大输入高宽（同时保存图像信息）***
![[fcn_1.png]]$Y[i:i+h,j:j+w]+=X[i,j]*K$(kernel)

# 模型
![[fcn_2.png]]
**核心思路：** 使用转置卷积代替CNN最后的全连接层，从而实现每个像素的预测
将转置后得到输出构建k个通道数（classification的数量,类似yolo的操作，将预测目标的标签存在通道上）（$224\times224\times7$），从而进行模型训练

# 代码
```python

import torch
import torch.nn as nn
from torchvision import models

class FCN32s(nn.Module):
    
    FCN-32s 模型结构
    使用预训练的VGG16作为特征提取器（Encoder）。
    
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        
        # 1. 编码器（Encoder）：基于预训练的VGG16
        # 我们使用VGG16的特征部分（features），并去除后面的全连接层。
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # VGG16特征提取部分，直到Pool5。
        # 在FCN中，我们将Pool5看作是特征提取的末端，其特征图尺寸是输入图像的 1/32。
        # 这一部分包含了卷积层和激活函数，用于提取图像的高级语义特征。
        self.features = vgg16.features
        
        # 2. 将VGG16的Pool5后的全连接层转换为卷积层 (Conv6 和 Conv7)
        # 这是FCN的关键步骤之一，将分类网络转换为全卷积网络。
        # 两个 1x1 卷积层用于将高维特征图映射到类别空间。
        
        # 原始VGG16的第一个全连接层 (4096 维)
        self.classifier = nn.Sequential(
            # Conv6: 替换 VGG16 的 FC6
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Conv7: 替换 VGG16 的 FC7
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Conv8 (Score Layer): 1x1 卷积，将特征维度映射到类别数
            # 这一层得到的是最终的语义分割预测图（score map）。
            nn.Conv2d(4096, num_classes, kernel_size=1, padding=0)
        )

        # 3. 解码器（Decoder）：反卷积层/转置卷积层
        # 使用转置卷积（nn.ConvTranspose2d）进行上采样，恢复特征图尺寸。
        # FCN-32s 只进行一次 32 倍上采样。
        self.upsample_32x = nn.ConvTranspose2d(
            in_channels=num_classes, 
            out_channels=num_classes, 
            kernel_size=64, # 64x64 的核进行32倍上采样
            stride=32,       # 步长为32
            padding=16,      # padding 确保输出尺寸正确
            bias=False
        )


    def forward(self, x):
        # 1. 编码器：特征提取
        x = self.features(x)  # 输出特征图尺寸为输入图像的 1/32
        
        # 2. 分类器（全卷积转换）
        x = self.classifier(x) # 输出 score map，通道数为 num_classes
        
        # 3. 解码器：32倍上采样，恢复到原始图像尺寸
        x = self.upsample_32x(x)
        
        return x

# --- 示例用法和模型测试 ---

# 假设我们需要在 PASCAL VOC 上进行分割，它有 21 个类别（20 个前景物体 + 1 个背景）。
NUM_CLASSES = 21 

# 实例化模型
model = FCN32s(num_classes=NUM_CLASSES)

# 模拟输入：Batch size=1, 3通道彩色图, 尺寸 224x224
# 实际分割中，输入尺寸通常更大，例如 512x512 或 480x480。
dummy_input = torch.randn(1, 3, 224, 224) 

# 模型前向传播
output = model(dummy_input)

print(f"输入尺寸: {dummy_input.shape}")
print(f"输出尺寸: {output.shape}") 
# 预期的输出尺寸应该是 (Batch, Num_Classes, Input_H, Input_W)，
# 即 (1, 21, 224, 224) 
# 注意：由于VGG的Pool层和ConvTranspose的参数设置，实际输出尺寸可能与输入尺寸略有不同，
# 在实际训练中需要进行裁剪（crop）或更精细的参数调整来保证尺寸完全匹配。
```

| **模块/操作**             | **PyTorch 对应代码**                  | **作用描述 (语义分割中的意义)**                                                                                                  |
| --------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **编码器 (Encoder)**     | `models.vgg16.features`           | **特征提取：** 逐步减小特征图的尺寸（如 $224 \to 112 \to 56 \to 28 \to 14 \to 7$），同时增加通道数，从图像中提取出**高级、抽象的语义特征**。它将像素级别的特征编码成低分辨率的特征图。 |
| **全连接层转换**            | `nn.Conv2d(..., kernel_size=7)`   | **全卷积化：** 将 VGG 等分类网络末尾的全连接层替换为卷积层，使网络能够接受**任意尺寸**的输入图像，这是 FCN 的核心。它将编码器提取的特征映射到一个高维的特征空间。                           |
| **得分层 (Score Layer)** | `nn.Conv2d(4096, num_classes, 1)` | **类别映射：** 使用 $1 \times 1$ 卷积将特征通道数压缩为所需的**类别数**（如 21）。其输出被称为 **Score Map**，每个像素点的通道值即为该像素属于对应类别的分数。                  |
| **解码器 (Decoder)**     | `nn.ConvTranspose2d`              | **上采样：** 使用**转置卷积 (Transpose Convolution)** 将低分辨率的 Score Map 重新放大到**原始输入图像的尺寸**。FCN-32s 只进行一次 32 倍上采样。               |
| **转置卷积**              | `kernel_size=64, stride=32`       | **尺寸恢复：** 是一种可学习的上采样方法，它将 $1/32$ 尺寸的特征图放大 $32$ 倍，从而为每个原始像素生成一个类别预测。                                                  |
| **损失函数** (训练时)        | `nn.CrossEntropyLoss`             | **优化目标：** 用于衡量模型输出的 Score Map 与**真实分割标签**之间的差异。在语义分割中，它通常作用于每个像素点，计算像素级别的分类损失。                                       |
| **Softmax** (推理时)     | `torch.softmax(output, dim=1)`    | **概率转换：** 将 Score Map 的分数转换为每个类别的**概率分布**，并选择概率最高的类别作为最终的分割结果。                                                       |
