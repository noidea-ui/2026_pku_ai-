# CIFAR-10 图像分类实验

## 1. 实验简介

本实验使用 PyTorch 实现 LeNet，并在 CIFAR-10 数据集上完成图像分类。训练过程使用 CUDA 加速，使用 W&B 记录 Batch 级别的 `loss` 和 Epoch 级别的 `epoch_loss`，最后统计测试集整体准确率以及 10 个类别各自的准确率。

## 2. 环境配置

- Python 3.10.21
- PyTorch
- torchvision
- CUDA
- wandb 0.30.0
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU

安装 W&B：

```bash
python -m pip install wandb
wandb login
```

PyTorch 和 torchvision 应根据本机的操作系统、Python 版本以及 CUDA 版本，从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应的安装命令。

## 3. 运行方式和测试：

```bash
python train.py
```

程序会自动读取或下载 CIFAR-10 数据集，并在训练开始前输出模型参数、输入图像和标签所在的设备。三者应均显示为 `cuda:0` 或其他 CUDA 设备。

## 4. 模型与训练设置

使用的模型为适用于 CIFAR-10 的 LeNet：

| 层 | 参数 |
| --- | --- |
| Conv1 | `3 -> 6`, kernel `5 x 5` |
| ReLU + MaxPool | kernel `2 x 2`, stride `2` |
| Conv2 | `6 -> 16`, kernel `5 x 5` |
| ReLU + MaxPool | kernel `2 x 2`, stride `2` |
| Fully Connected | `400 -> 120 -> 84 -> 10` |

训练设置如下：

| 参数 | 设置 |
| --- | --- |
| Batch size | `128` |
| Epochs | `10` |
| Loss function | Cross Entropy Loss |
| Optimizer | SGD |
| Learning rate | `0.01` |
| Momentum | `0.0`（当前代码） |
| Input normalization | mean=`(0.5, 0.5, 0.5)`, std=`(0.5, 0.5, 0.5)` |
























## 5. W&B 记录



代码记录了以下指标：

- `loss`：每个训练 Batch 的损失；
- `epoch_loss`：每个 Epoch 的平均训练损失；
- `train_accuracy`：每个 Epoch 的训练准确率；
- `test_loss`：测试集平均损失；
- `test_accur 6&B Run 链接]` 替换为 W&B 页面中的实际链接，并确认 `[填写实际值]` 与对应实验使用的 Momentum 一致。

## 6. 结论

本实验完成了 CIFAR-10 数据集的数据加载、LeNet 模型构建、CUDA 训练、W&B 记录以及测试评估。测试阶段不仅计算了整体准确率，还按照真实类别分别统计了 10 个类别的准确率。不同类别之间的识别效果存在差异，其中当前记录中 automobile 和 frog 的准确率相对较高，而 bird、deer 和 cat 的准确率相对较低，说明简单的 LeNet 对不同类别的特征提取能力并不相同。
