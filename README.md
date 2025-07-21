# PyTorch实践：从零构建、训练到部署一个MNIST分类器

## 1. 项目概述

这是我个人学习和实践`PyTorch`深度学习框架的一个入门项目。

项目的核心目标是，完整地走一遍使用`PyTorch`进行神经网络开发的标准工作流。我选择经典的**MNIST手写数字识别**任务，从零开始，实现了**数据加载、模型构建、模型训练、保存与加载、以及最终的图片推理**的全过程。

此外，我还额外探索了`PyTorch`的一个高级特性——**`torch.fx`**，学习了如何对模型的计算图进行符号追踪和动态修改。

---

## 2. 核心实践内容

### **a. 基于`PyTorch`的MNIST分类器 (`test.py`, `numberlize.py`)**

这是一个功能完备的深度学习项目，包含了以下几个关键环节：

*   **模型架构 (`MyModule`):**
    *   使用`torch.nn.Module`构建了一个经典的三层**全连接神经网络 (MLP)**。
    *   `28*28 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Logits)`

*   **数据处理:**
    *   使用`torchvision.datasets.MNIST`加载标准数据集。
    *   通过`torchvision.transforms`对输入图像进行**ToTensor**和**Normalize**的预处理。
    *   利用`torch.utils.data.DataLoader`来创建数据加载器，实现数据的**批量化(Batching)**和**随机打乱(Shuffle)**。

*   **模型训练 (`test.py`):**
    *   实现了一个标准、完整的训练循环。
    *   **优化器:** `torch.optim.Adam`
    *   **损失函数:** `nn.CrossEntropyLoss` (适用于多分类任务)
    *   训练流程完整覆盖了**梯度清零、前向传播、计算损失、反向传播和参数更新**五大核心步骤。
    *   训练结束后，使用`torch.save()`将模型的**状态字典 (state_dict)** 保存到文件。

*   **模型推理 (`numberlize.py`):**
    *   这是一个独立的脚本，用于演示如何**加载已训练好的模型权重**。
    *   它接收一张本地的数字图片 (`test9.png`)，通过`Pillow`库进行读取和预处理，然后用加载好的模型对其进行分类预测。

### **b. 对`torch.fx`的探索 (`fx_demo.py`)**

在掌握了基本的模型使用后，我希望进一步理解`PyTorch`的底层机制。`torch.fx`是一个强大的工具，用于对`nn.Module`进行函数变换。

*   **符号追踪 (Symbolic Trace):**
    *   我使用`torch.fx.symbolic_trace()`来捕获`MyModule`的**前向传播计算图 (Graph)**。
    *   这个过程将模型的执行流，从Python代码，转换成了一个**与设备无关的、可以被程序化分析和修改的中间表示 (Intermediate Representation, IR)**。

*   **计算图的动态修改:**
    *   我编写了一个`transform`函数，它接收一个模型，获取其计算图，然后**动态地在图中插入一个新的操作节点**（例如，在模型的最后一层`layer3`之后，插入一个`torch.relu`调用）。
    *   这展示了`torch.fx`在**模型重构和优化**方面的巨大潜力。

---

## 3. 技术与能力总结

*   **深度学习框架:** 熟练掌握了`PyTorch`的核心API，能够独立完成**数据处理、模型构建、训练和部署**的全链路开发。
*   **神经网络基础:** 理解全连接层、激活函数、损失函数和优化器等基本概念，并能将其应用于实际的分类任务。
*   **框架底层探索:** 对`torch.fx`等高级特性的学习，表明我具备**深入理解框架底层原理**的意愿和能力。

---

## 4. 如何运行

1.  克隆本仓库到本地。
2.  安装所有依赖库：
    ```bash
    pip install torch torchvision pillow matplotlib
    ```
3.  **训练模型:**
    ```bash
    python test.py
    ```
    (这将会下载MNIST数据集，并训练模型，最终在当前目录下生成`model_state_dict.pth`文件)

4.  **进行图片推理:**
    *   确保当前目录下有一张名为`test9.png`的、背景为黑色的手写数字图片。
    *   运行推理脚本：
    ```bash
    python numberlize.py
    ```
5.  **探索`torch.fx`:**
    ```bash
    python fx_demo.py
    ```