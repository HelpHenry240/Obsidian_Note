这部分主要分为两个板块：**可视化（看看训练效果如何）** 和 **模型保存与加载（把训练好的模型存下来）**。

---

### 第一板块：可视化训练结果

最直观的方法就是画出 **Loss（损失）曲线** 和 **Accuracy（准确率）曲线**。如果 Loss 一直在下降，说明模型在学习；如果 Accuracy 越来越高，说明模型越来越准。

#### 1. 如何收集数据？
我们需要在训练循环中创建一个列表（List），把每个 Epoch 的 Loss 或 Accuracy 存起来。

```python
import matplotlib.pyplot as plt

# 1. 定义容器
loss_history = []  # 存放每个epoch的loss
acc_history = []   # 存放每个epoch的准确率

for epoch in range(num_epochs):
    # ... (这里是训练代码) ...
    
    # 假设 epoch_loss 是当前这一轮的平均 loss
    loss_history.append(epoch_loss) 
    
    # 假设 epoch_acc 是当前这一轮的验证集准确率
    acc_history.append(epoch_acc)
    
    print(f"Epoch {epoch}: Loss={epoch_loss}, Acc={epoch_acc}")
```

#### 2. 如何画图？
使用 Python 最常用的绘图库 `matplotlib`。

```python
# 画 Loss 曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() # 显示图片
```

*   **观察重点**：
    *   **正常**：曲线一开始下降很快，后来趋于平缓。
    *   **过拟合 (Overfitting)**：训练 Loss 还在下降，但验证 Loss 开始上升（如果你画了验证 Loss 的话）。
    *   **不收敛**：Loss 震荡得很厉害，或者根本不下降（可能是学习率太大）。

---

### 第二板块：模型的保存与加载 (Save & Load)

在 PyTorch 中，保存模型主要保存的是**参数（Weights & Biases）**，也就是我们常说的“权重”。这些参数存在一个叫 `state_dict` 的字典里。

#### 1. 保存模型 (Saving)

通常有两种保存需求：

**场景 A：训练完成，只保存模型参数用于预测（推荐）**
这是最节省空间的方式，只保存参数，不保存模型的代码结构。

```python
# 假设 model 是你训练好的模型
save_path = "my_best_model.pth" # 后缀通常用 .pth 或 .pt

# model.state_dict() 获取所有参数字典
# torch.save() 将这个字典序列化保存到硬盘
torch.save(model.state_dict(), save_path)

print("模型参数已保存！")
```

**场景 B：训练未完成，保存检查点 (Checkpoint) 以便稍后继续训练**
如果训练要跑几天，你需要每隔几个 Epoch 存一次，以防断电。这时不仅要存模型参数，还要存**优化器的状态**（因为优化器里也有动态调整的学习率等信息）和当前的 Epoch。

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),      # 模型参数
    'optimizer_state_dict': optimizer.state_dict(), # 优化器参数 (比如动量信息)
    'loss': running_loss
}

torch.save(checkpoint, "checkpoint_epoch_50.pth")
```

---

#### 2. 加载模型 (Loading)

加载和保存是对应的。

**场景 A：加载参数进行预测 (Inference)**

```python
# 1. 必须先实例化一个结构完全一样的模型对象
model = MyModel() 

# 2. 加载硬盘上的参数字典
# torch.load 把文件读进来
# map_location='cpu' 是为了防止你是在 GPU 上训练的，但在没有 GPU 的电脑上加载
state_dict = torch.load("my_best_model.pth", map_location='cpu')

# 3. 将参数填入模型
model.load_state_dict(state_dict)

# 4. !关键一步!：切换到评估模式
model.eval() 

# 接下来就可以用 model(input) 去预测了
```

*   **为什么一定要 `model.eval()`？**
    *   如果你不加这行，Dropout 依然会随机丢弃神经元，BatchNorm 依然会计算当前 Batch 的均值，导致预测结果每次都不一样且不准确。

**场景 B：加载检查点继续训练 (Resume Training)**

```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 加载检查点
checkpoint = torch.load("checkpoint_epoch_50.pth")

# 恢复模型参数
model.load_state_dict(checkpoint['model_state_dict'])

# 恢复优化器参数 (这步很重要，否则学习率策略会重置)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 恢复 Epoch
start_epoch = checkpoint['epoch']

# 切换回训练模式
model.train() 

# 从断点处继续循环
for epoch in range(start_epoch, num_epochs):
    # ... 继续训练
```

---

### 总结：各个函数的作用

1.  **`model.state_dict()`**：
    *   **作用**：把模型里所有的权值（Weights）和偏置（Bias）打包成一个 Python 字典。
    *   **比喻**：相当于把学生的脑子里的知识点全部提取出来，写在一本书上。

2.  **`torch.save(obj, path)`**：
    *   **作用**：把对象保存到硬盘文件。
    *   **比喻**：把书存档。

3.  **`torch.load(path)`**：
    *   **作用**：把硬盘文件读取到内存。
    *   **比喻**：把书取出来。

4.  **`model.load_state_dict(loaded_dict)`**：
    *   **作用**：把读取到的参数字典，“灌输”给当前的模型对象。
    *   **比喻**：把书里的知识点塞进一个新学生的脑子里，让他拥有和之前那个学生一样的能力。

5.  **`map_location='cpu'` (在 load 时使用)**：
    *   **作用**：处理设备不匹配问题。如果你在 RTX 4090 上训练的模型（参数在 CUDA 上），想在自己的笔记本（CPU）上跑，必须加这个参数，否则会报错。