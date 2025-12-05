# 核心框架
数据是深度学习的“燃料”，你对燃料的理解直接决定了引擎（模型）能跑多快、多远。
### 第一层：核心概念与机制（必须掌握）
这是地基，不懂这些无法进行任何实验。

1.  **数据集划分（Splitting）：**
    *   必须深刻理解 **训练集 (Train)**、**验证集 (Validation)** 和 **测试集 (Test)** 的区别和作用。
    *   **重点：** 为什么要划分？什么是“信息泄露（Data Leakage）”？为什么不能在测试集上调参？
2.  **数据表现形式（Tensor Representation）：**
    *   知道数据在代码中是如何存在的。
    *   **CV（计算机视觉）：** 图像通常是 `[Batch_Size, Channels, Height, Width]` (PyTorch) 或 `[Batch, H, W, C]` (TensorFlow)。像素值范围是 0-255 还是 0-1？
    *   **NLP（自然语言处理）：** 文本如何变成 Token ID，Embedding 是什么，Padding 和 Mask 的作用。
3.  **数据加载机制（Dataloader）：**
    *   在 PyTorch 中，必须掌握 `Dataset` 类（如何读取单个样本）和 `DataLoader` 类（如何批处理、打乱数据、多线程读取）。
    *   理解 **Batch Size** 对显存和收敛速度的影响。

### 第二层：领域的“通用语言”（了解即可，但这不仅是数据）
科研圈交流时，大家会用经典数据集作为衡量指标。你不需要下载它们，但要“听说过”。

1.  **CV 领域：**
    *   **MNIST / Fashion-MNIST：** 模型的“Hello World”，用来调试代码跑通流程。
    *   **CIFAR-10 / CIFAR-100：** 轻量级学术基准。
    *   **ImageNet (ILSVRC 2012)：** 深度学习爆发的源头，衡量模型提取特征能力的“金标准”。
    *   **COCO：** 目标检测和分割的标杆。
2.  **NLP 领域：**
    *   **IMDB / SST-2：** 情感分类入门。
    *   **SQuAD：** 问答系统。
    *   **WMT：** 机器翻译。
    *   **GLUE / SuperGLUE：** 衡量大模型（BERT/GPT系列）综合能力的榜单。

**在这个阶段，你需要知道：** 你的研究方向通常用哪个数据集做 Baseline（基线）？SOTA（当前最佳效果）通常是在哪个数据集上跑出来的？

### 第三层：数据预处理与增强（核心技能）
这决定了模型效果的上限。

1.  **预处理（Preprocessing）：**
    *   **归一化（Normalization）：** 为什么要把图像像素减去均值除以方差？（加速收敛）。
    *   **Resize / Crop：** 怎么把不同尺寸的图变成一样大输入网络。
    *   **Tokenization（NLP）：** BPE, WordPiece 是什么。
2.  **数据增强（Data Augmentation）：**
    *   **CV：** 随机翻转、旋转、颜色抖动、CutMix、MixUp。知道这些是为了防止**过拟合（Overfitting）**。
    *   **NLP：** 随机掩码（Masking）、回译（Back-translation）。

### 第四层：科研思维（进阶认知）
当你开始看论文或做项目时，需要关注以下几点：

1.  **数据偏差（Bias）与分布（Distribution）：**
    *   **OOD（Out-of-Distribution）：** 训练数据和测试数据分布不一致会发生什么？这是目前科研的热点。
    *   **长尾分布（Long-tail）：** 大部分类别样本很多，少部分类别样本很少，怎么处理？
2.  **评价指标（Metrics）与数据的关系：**
    *   数据集类别不平衡（Imbalance）时，为什么不能只看 Accuracy？（要看 F1-score, AUC, mAP）。
3.  **License（版权）：**
    *   做学术虽然相对宽松，但要意识到某些数据集（如 WebFace）因为隐私或版权问题被撤下的情况。商业使用必须看 License (CC-BY, MIT, Apache 等)。

---

### 总结：如何检验自己是否“达标”？

如果你能回答以下 3 个问题，说明你在入门阶段关于数据集的知识已经合格了：

1.  **实操题：** 给你一个文件夹的图片和对应的 CSV 标签文件，你能写出一个 PyTorch 的 `CustomDataset` 类，并用 `DataLoader` 把它读出来喂给模型吗？
2.  **概念题：** 如果模型在训练集上 Loss 很低，但在验证集上 Loss 很高，这说明了什么？你会尝试用什么数据增强手段来解决？
3.  **视野题：** 在你感兴趣的那个细分领域（比如人脸识别、文本摘要、医学影像），大家公认最难、最权威的 1-2 个数据集叫什么名字？

**用时再查，重在理解 pipeline（流水线）和 evaluation（评估标准）。**

在使用 PyTorch 进行科研时，处理数据通常遵循 **“Dataset（定义数据） -> Transforms（变换数据） -> DataLoader（加载数据）”** 这一标准流水线。

以下是你必须掌握的核心代码模块和函数，按实际开发流程排序：

---

# Coding
### 第一阶段：定义数据 (Dataset)

这是最底层、最核心的部分。你需要继承 `torch.utils.data.Dataset` 类来告诉 PyTorch 你的数据在哪里、长什么样。

#### 1. 核心类与方法
必须掌握如何重写以下三个“魔法方法”：

*   **`__init__(self, ...)`**: 初始化。
    *   **要做的事：** 传入文件路径、标签列表、Transform（变换）。通常在这里读取 CSV 文件或获取所有图片的文件名列表。
    *   **切记：** **不要**在这里把所有图片都读进内存（除非数据量极小），只存路径！否则内存会爆炸。
*   **`__len__(self)`**: 告诉 PyTorch 数据集有多大。
    *   **代码：** `return len(self.data_list)`
*   **`__getitem__(self, index)`**: 核心中的核心。
    *   **要做的事：** 根据索引 `index`，读取**单张**图片/文本，进行预处理，并返回 `(data, label)`。

#### 2. 代码模板 (背诵级别)
这是一个处理图像数据的标准模板：

```python
from torch.utils.data import Dataset
import os
from PIL import Image

class MyCustomDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 假设 labels_file 是一个列表，存着 [文件名, 类别]
        self.img_labels = ... # 读取csv或txt逻辑
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 1. 获取路径
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        # 2. 读取数据 (PIL读取是RGB)
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx][1]
        
        # 3. 应用变换 (转Tensor, 归一化等)
        if self.transform:
            image = self.transform(image)
            
        # 4. 返回样本和标签
        return image, label
```

---

### 第二阶段：数据变换与增强 (Transforms)

数据读进来是图片（PIL）或数组（NumPy），模型吃的是 Tensor。这一步负责转换。

#### 1. 核心库
`torchvision.transforms` (CV领域)

#### 2. 必会函数
*   **`transforms.Compose([...])`**: 将多个操作串联起来的容器。
*   **`transforms.ToTensor()`**: **最重要**。
    *   作用：将 PIL Image 或 NumPy `(H, W, C)` (0-255) 转为 Tensor `(C, H, W)` (0.0-1.0)。这是 PyTorch 模型的标准输入格式。
*   **`transforms.Normalize(mean, std)`**: 归一化。
    *   作用：`(input - mean) / std`。加速收敛。ImageNet 的通用均值方差是 `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`。
*   **`transforms.Resize((h, w))`**: 强制把图片缩放到统一大小（这就不用担心显存忽大忽小）。

#### 3. 代码示例
```python
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 统一大小
    transforms.RandomHorizontalFlip(),  # 数据增强：随机翻转
    transforms.ToTensor(),              # 转 Tensor (0-1)
    transforms.Normalize(mean=[...], std=[...]) # 归一化
])
```

---

### 第三阶段：加载数据 (DataLoader)

Dataset 只是定义了一个样本怎么读，`DataLoader` 负责把它们打包成 Batch，利用多进程加速读取。

#### 1. 核心类
`torch.utils.data.DataLoader`

#### 2. 关键参数 (必须理解)
*   **`dataset`**: 把刚才写的 `MyCustomDataset` 实例传进去。
*   **`batch_size`**: 一次喂给模型多少数据（如 32, 64）。显存不够就调小这个。
*   **`shuffle`**: **训练集必须设为 `True`**（打乱顺序，防止模型记住顺序），验证/测试集设为 `False`。
*   **`num_workers`**: 多少个子进程在后台读数据。
    *   Windows 上通常设为 0（主进程），Linux 上可以设为 4, 8, 16。设太大反而慢。
*   **`drop_last`**: 如果数据总数除不尽 Batch Size，是否丢弃最后剩下的那几个。训练时通常 `True`（怕影响 BatchNorm），测试时 `False`。

#### 3. 进阶参数：`collate_fn` (选修)
*   当你处理**变长数据**（如 NLP 里的句子长短不一，或目标检测里一张图有 3 个框，另一张有 5 个框）时，默认的打包方式会报错。
*   你需要自己写一个 `collate_fn` 函数来告诉 DataLoader 如何把这些长短不一的数据拼成一个 Batch（通常涉及 Padding）。

#### 4. 代码示例
```python
from torch.utils.data import DataLoader

train_dataset = MyCustomDataset(..., transform=data_transform)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)
```

---

### 第四阶段：实战中的“偷懒”技巧 (常用)

科研中为了快速验证，有一些现成的工具不需要手写 Dataset。

#### 1. `ImageFolder` (CV 神器)
如果你的图片是按文件夹分类放好的（例如 `dog/xxx.jpg`, `cat/xxx.jpg`），直接用这个，不用写 `__init__` 和 `__getitem__`。
```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder(root="path/to/data", transform=data_transform)
# dataset.classes 会自动识别 ['cat', 'dog']
```

#### 2. `random_split` (数据集划分)
如果你只有一个总的数据集，想手动分出训练集和验证集：
```python
from torch.utils.data import random_split

# 假设 dataset 有 1000 张图，想 8:2 分
train_set, val_set = random_split(dataset, [800, 200])
```

---

### 第五阶段：训练循环中的数据操作

最后，在 `train.py` 的循环里，你只涉及这两行关键代码：

1.  **解包**：从 Loader 里拿数据。
2.  **搬运**：把数据从 CPU 搬到 GPU。

```python
# 假设 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for batch_idx, (inputs, targets) in enumerate(train_loader):
    # 1. 搬运到 GPU (非常重要，否则报错)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # 2. 喂给模型
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # ... 反向传播 ...
```

---

### 总结：你需要检查自己是否掌握的清单

1.  [ ] 能否默写出 `Dataset` 类的 `__init__`, `__len__`, `__getitem__` 结构？
2.  [ ] 知道 `ToTensor()` 是把 `(H,W,C)` 变成 `(C,H,W)` 且归一化到 0-1 吗？
3.  [ ] 知道 `DataLoader` 里的 `shuffle=True` 应该用在训练集还是测试集？
4.  [ ] 知道代码报错 `Expected object of device type cuda but got device type cpu` 时，通常是因为忘了写 `.to(device)` 吗？

