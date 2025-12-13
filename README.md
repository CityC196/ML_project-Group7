# 机器学习基础 (2025) - 课程大作业 (第七组)

> **北京大学工学院 2025 年《机器学习基础》课程大作业代码仓库**

本仓库包含第七组关于专家模型复现、重训练以及 Router 模块训练的相关代码与数据集。

## 👥 小组成员
**石宇宸、龙想、田浩远、李宗远、石昊**

---

## 📂 目录结构说明

本仓库主要包含以下两个核心模块：

### 1. `test_originExpert` (官方模型复现)
用于验证和复现官方提供的专家模型性能。
* **Reproduce_Code**: 复现实验的核心代码。
* **Results_Origin**: 运行官方代码生成的原始输出数据。
* **Visualization**: 官方代码预测结果与原始数据集的对比可视化图表。

### 2. `train_ownExpert` (模型重训练)
包含针对 Unet 和 PINN 模型的降维重训练代码，其中数据集的空间降维方式均为均匀采样。
* **FNO 模型**: 由于采用自适应结构，**无需**重新训练，最终输入为维度为`[10 * 64]` 。
* **Unet 模型**: 在不同数据集下进行了降维重训练，输入维度配置如下：
    * `diff-sorp` 数据集: `[4 * 256]` 
    * `advection` 数据集: `[4 * 256]`
    * `3d-cfd` 数据集: `[3 * 8 * 8 * 8]`
* **PINN 模型**: 训练维度配置与上述 Unet 相同，并在 `Sample0` 上进行训练。

---

## 🔗 资源下载 (Models & Datasets)

为了方便复现，我们将训练好的模型权重、生成数据集及 Router 数据集托管在 Hugging Face 社区，同时提供夸克云盘作为国内下载备用。

| 资源名称 | 内容说明 | Hugging Face 下载 | 夸克云盘下载 (备用) |
| :--- | :--- | :--- | :--- |
| **专家模型权重**<br>(Expert Models) | 包含 `ExpertModels_official` (官方模型) 与 `ExpertModels_own` (降维重训练模型及对比图)。 | [点击跳转](https://huggingface.co/CrisC196/ExpertModels) | [点击下载](https://pan.quark.cn/s/ea1805d61471) |
| **文生计算数据集**<br>(Text-to-Computation) | 包含用于文生模块的数据集及其生成代码。 | [点击跳转](https://huggingface.co/datasets/CrisC196/TEXT_to_Computation) | [点击下载](https://pan.quark.cn/s/09f16adf09d3) |
| **Router 数据集**<br>(Router Dataset) | 用于训练 Router 判别模块的数据集。 | [点击跳转](https://huggingface.co/datasets/CrisC196/Router_dataset) | [点击下载](https://pan.quark.cn/s/87ec304a34fc) |

---
