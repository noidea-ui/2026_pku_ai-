# HW3：CUDA 神经网络基础模块

本工程实现《人工智能中的编程》第三次作业要求的五类模块核心函数：FC、Conv2D、Max Pooling、SoftMax、Cross Entropy。代码使用单精度 `float`、CUDA C++、cuBLAS 和 cuRAND，不依赖 PyTorch 等深度学习框架。

> 当前交付重点是核心实现和必要测试；作业报告暂未编写。当前执行环境不包含 Windows CUDA 13.1 + MSVC 工具链，因此本包尚未在你的本机 GPU 上完成真实编译/运行验证。测试代码已包括 CPU 参考实现和梯度有限差分检查，须在目标环境运行后确认结果。

## 工程结构

```text
HW3/
├─ include/
│  ├─ common.cuh
│  ├─ fc.cuh
│  ├─ conv2d.cuh
│  ├─ max_pool.cuh
│  ├─ softmax.cuh
│  └─ cross_entropy.cuh
├─ src/
│  ├─ fc.cu
│  ├─ conv2d.cu
│  ├─ max_pool.cu
│  ├─ softmax.cu
│  ├─ cross_entropy.cu
│  └─ main.cu
├─ tests/
│  └─ test_utils.h
├─ build.ps1
└─ README.md
```

## 环境要求

- Windows 11（或兼容的 Windows 环境）
- CUDA Toolkit 13.1，包含 `nvcc`、cuBLAS、cuRAND
- Visual Studio 2022 C++ 桌面开发工具及 x64 MSVC 编译器
- 支持所选 CUDA 架构的 NVIDIA GPU

默认构建目标是 `sm_120`，对应本工程预期使用的 RTX 5060。若在其他 GPU 上编译，可在构建前设置架构，例如：

```powershell
$env:HW3_CUDA_ARCH = "sm_XX"
```

请把 `sm_XX` 替换为实际 GPU 对应的架构标记。也可以按需修改 `build.ps1` 中的默认值。

## 编译与运行

推荐从 **x64 Native Tools Command Prompt for VS 2022** 启动 VS Code，以确保 `cl.exe`、`nvcc` 和 CUDA 库环境可用：

```bat
cd /d D:\path\to\HW3
code .
```

在 VS Code 终端中检查：

```powershell
nvcc --version
where.exe cl
```

然后编译并运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\build\hw3.exe
```

脚本使用 `nvcc -std=c++17` 构建全部 `.cu` 文件，并链接 cuBLAS/cuRAND。代码不启用 `--use_fast_math`，以免降低数值梯度检查的可解释性。

## 模块约定

- FC：`X[N,Cin]`、`W[Cout,Cin]`、`b[Cout]`、`Y[N,Cout]`，逻辑上按行主序存储。
- Conv2D：输入和输出采用 NCHW；权重是 `[Cout,Cin,3,3]`；zero padding=1，stride=1。`im2col` 缓冲按逻辑 `[Cin*9,H*W]` 的列主序存储，以便与 cuBLAS GEMM 配合。
- Max Pooling：`2x2` 窗口、stride=2，要求输入高宽为正偶数；mask 按作业附录接口使用 `float*`，保存最大值对应的输入绝对线性索引；反向传播时转换为整数索引。
- SoftMax：对 `[N,C]` 每行独立计算，减去该行最大值以增强数值稳定性。
- Cross Entropy：前向返回 batch 平均损失；反向返回对 logits 的合并梯度 `(p - one_hot(label))/N`。
- 参数初始化：测试通过 cuRAND 以固定种子初始化 FC/Conv 的权重和偏置；输入与上游梯度使用确定性测试数据。

## 测试内容

`main.cu` 会依次执行以下测试，并对任何失败返回非零退出码：

1. FC 正向及 `dX/dW/db` 与 CPU 参考实现比较；选定元素进行有限差分梯度检查。
2. `im2col` 的列布局和 zero padding；`col2im` 重叠贡献累加。
3. Conv2D 正向及 `dX/dW/db` 与朴素 CPU 卷积比较；选定元素进行有限差分梯度检查。
4. Max Pooling 输出值、mask 与反向梯度。
5. SoftMax 与 CPU 参考结果比较，并检查大幅度 logits 下每行概率和。
6. Cross Entropy 平均损失、合并梯度及 logits 有限差分检查。

测试阈值是浮点数容差，而不是要求 CPU/GPU 结果逐位相同。请以目标机器实际运行的输出为最终测试结果，不要在未运行前将测试状态写成“全部通过”。

## 接口限制与注意事项

- 核心函数的输入/输出指针应指向 GPU device memory。
- 调用 FC/Conv2D 时必须传入已经创建的 `cublasHandle_t`。
- 调用方负责准备显存；Conv2D 函数内部会分配和释放临时 `im2col` 缓冲区。
- 当前实现面向作业指定的 3x3 卷积、zero padding、stride 1，不覆盖 depthwise、dilated 或 transposed convolution。
- Pooling 的 `mask` 按附录 B 的 `float* out_mask` 接口实现，以兼容作业给出的接口样式。索引以 float 形式保存，在本作业的小规模张量下可精确表示。
