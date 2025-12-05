[paper](https://arxiv.org/pdf/1506.02640)
# 预测阶段（forward:不进行参数更新）
![[cnn model.png]]
**24层卷积层提取特征；2层MLP回归得到$7 \times 7 \times 30$ 的tensor**
## 基本概念

**grid cell：** $7 \times 7$ 是因为yolo将图片分成若干个网格，总共网格数就是$7\times7$，每一个grid cell能够预测b(yolo中是2)个bouding boxes，与Ground Truth IoU 最大的bounding box （用粗线表示）负责预测这个物体，且一个grid cell只能预测一个物体
**bounding boxes:** 大小不定，但中心是该grid cell,是一个包含（x,y,h,w,c：xy是中心点坐标；hw是框的高度和宽度；c是置信度(Ground Truth)：检测到物体的概率的张量
**Conditioned on object:** P(Car|Object)
![[yolov1_1.png]]
**$7 \times 7 \times 30$ 的tensor:** $7 \times 7$ 指的是grid cell的数量，一个grid cell是一个30维张量，有以下三个部分组成：
1. bounding box1:$x_1,y_1,h_1,w_1,c_1$，5个参数
2. bounding box2:$x_2,y_2,h_2,w_2,c_2$，5个参数
3. **voc(数据集)**：包含20个类别（classes）的概率，P(obj is $class_i$|obj is in box)
**$7\times7\times30=1470$**
## 后处理（NMS非极大值抑制）
**目的：将$7\times7\times30$的张量变为最后的处理结构，并把低置信度和高度重合的框去掉**
 ![[yolov1_3.png]]
1. 分别将20的类别的概率与两个bounding boxes的置信度相乘:
$P(Obj Is Class_i And Obj Is In Box)=P(Obj Is In Box)(c_i) \times P(Obj Is Class_i|Obj Is In Box)$
从而得到两个20维的张量，最终49个grid cell将会得到98个20维的张量。
2. 进行后处理：
![[yolov1_4.png]]
     1. **将第一个类别小于threshold(阈值)的清零，并重新排序把非零张量放在前面**
     2. 
 ![[yolov1_5.png]]
NMS思路：对从大到小排序的数据，取首个为对比对象，依次从第二个开始比较，如果IoU(bounding box重合程度)>thred(阈值)，则清零该框，否则向后遍历；当第一次遍历结束后，则取此时第二大概率的为对比对象，继续向后比较，直至全部非零数据作为对比对象遍历完。
       3. 对其他类别重复上述操作，直至遍历完所有类别（总共20次NMS）
3. 最后剩下的不全为0的张量就是最终结果，并将结果可视化在图片上

## 缺点
因为一个grid cell 只能预测一个物体，所以yolov1最多只能预测49个物体，这导致他在预测小目标时的效果不好




# 训练阶段（反向传播）

## loss function
**Ground Truth:** 人工标注的真实的检测框（中心就是该grid cell的位置），训练损失函数的目的就是让bounding box去逼近这些框（其中每个girdcell的两个bounding boxes中选取与ground truth IoU最大的），如果该gridcell的没有groudtruth标注，则舍弃它。
**核心处理的数据：** 中心点定位($x_i,y_i$),bounding box宽高($h_i,w_i$),负责检测物体的bbox的置信度($c_i->c_t(计算该框与groundtruth的IoU)$),不负责检测物体的bbox的置信度（$c_i->0$），类别预测误差($p_i$)
[算法精讲-训练阶段_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV15w411Z7LG?spm_id_from=333.788.videopod.episodes&vd_source=80bbbe03553b28a3cf02b9347c035eea&p=6)
![[yolo_loss function.png]]


# 代码


YOLOv1 的原始论文使用了 24 个卷积层（Conv Layers）和 2 个全连接层（FC Layers）。在 PyTorch 中，我们可以将其构建为一个 `nn.Module` 类。

```python
import torch
import torch.nn as nn

# 定义一个简化的 YOLOv1 模型类
class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        # S (split_size) = 7, B (num_boxes) = 2, C (num_classes) = 20 (for PASCAL VOC)
        super(YOLOv1, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # 1. 特征提取网络 (Backbone)
        self.feature_extractor = self._create_conv_layers()
        
        # 2. 预测头 (Prediction Head) - 全连接层
        self.fcs = self._create_fcs()

    def forward(self, x):
        # 1. 提取特征
        x = self.feature_extractor(x)
        
        # 2. 展平 (Flatten) - 将特征图展平为向量
        # (N, C, H, W) -> (N, C*H*W)
        x = torch.flatten(x, start_dim=1)
        
        # 3. 预测
        x = self.fcs(x)
        
        # 4. reshape to (N, S*S*(B*5 + C)) for final output (optional)
        return x 
```

---

### 2. 详细组件及作用 (Detailed Components and Roles)

|**PyTorch 模块**|**YOLOv1 概念**|**作用 (Role)**|**细节 (Details)**|
|---|---|---|---|
|`self.feature_extractor`|**特征提取网络** (Backbone)|从输入图像中提取高级语义特征，通常使用大量的 **卷积层 (Conv)** 和 **最大池化层 (MaxPool)**。|原始 YOLOv1 使用了类似 GoogleNet 的结构，包含 24 个卷积层，用于将 $448 \times 448$ 的输入图像降采样到 $7 \times 7 \times 1024$ 的特征图。|
|`Conv + ReLU + MaxPool`|**卷积层 + 激活函数 + 池化层**|标准 CNN 结构，用于**提取和压缩**图像信息。|YOLOv1 在许多卷积层后使用 Leaky ReLU 作为激活函数，在网络的末尾使用最大池化进行空间降维。|
|`torch.flatten`|**展平操作** (Flatten)|将特征提取网络输出的 $S \times S \times 1024$ 的三维特征图**转换为一维向量**，以便输入全连接层。|将 $N \times 1024 \times 7 \times 7$ 的张量展平为 $N \times (1024 \times 7 \times 7)$。|
|`self.fcs`|**预测头** (Prediction Head)|承担主要的**预测任务**，将展平后的特征映射到最终的输出维度。|通常包含两个全连接层。**第一个 FC 层**用于特征映射；**第二个 FC 层**的输出维度是 $S \times S \times (B \times 5 + C)$。|
|`Output` $\rightarrow S \times S \times (B \times 5 + C)$|**最终输出张量** (Final Tensor)|包含所有网格、所有边界框和所有类别的预测信息。|**$S \times S$**：图像被划分为的网格数 (e.g., $7 \times 7 = 49$)。<br><br>  <br><br>**$B \times 5$**：每个网格预测的 B 个边界框信息 (e.g., $2 \times 5$ = 10)。每个框包含 (x, y, w, h, Confidence)。<br><br>  <br><br>**$C$**：每个网格的类别预测 (e.g., 20)。|

### 3. PyTorch 代码实现细节 (PyTorch Implementation Details)

#### A. 简化的特征提取网络 (`_create_conv_layers`)

YOLOv1 的特征提取网络非常深。以下是一个**简化的结构**来表示其功能：

```python
    def _create_conv_layers(self):
        # 这是一个简化的表示，用于演示结构。原始YOLOv1结构更复杂。
        return nn.Sequential(
            # 示例：448x448x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ... 多个卷积/池化层堆叠 ... (降到 7x7)

            # 示例：最后几层，将通道数提升到 1024
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # 最终输出特征图： N x 1024 x 7 x 7 (假设 S=7)
        )
```

#### B. 预测头 (`_create_fcs`)

预测头负责将特征图映射到最终的预测结果。

```python
    def _create_fcs(self):
        # 展平后的输入维度：S * S * C_feat (e.g., 7 * 7 * 1024)
        flatten_size = self.S * self.S * 1024 
        # 输出维度：S * S * (B * 5 + C) (e.g., 7 * 7 * (2 * 5 + 20) = 1470)
        output_size = self.S * self.S * (self.B * 5 + self.C) 
        
        return nn.Sequential(
            # 第一个全连接层
            nn.Linear(flatten_size, 4096),
            nn.LeakyReLU(0.1),
            # 引入 Dropout 来防止过拟合 (可选)
            nn.Dropout(0.5), 
            
            # 第二个全连接层 (输出层)
            nn.Linear(4096, output_size) 
            # 注意：这里不再使用激活函数，因为输出需要是原始的边界框坐标、置信度和 logits。
        )
```

---

