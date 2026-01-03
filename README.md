# NanoMLP Spectrum Viz: 基于 JavaFX 的神经网络可解释性可视化

[![Java](https://img.shields.io/badge/Java-1.8-orange)](https://www.oracle.com/java/)
[![JavaFX](https://img.shields.io/badge/GUI-JavaFX-blue)](https://openjfx.io/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/Training-PyTorch-red)](https://pytorch.org/)

> 《AI辅助程序设计》课程实验项目 | 2025-2026学年第一学期

## 📖 项目简介 (Introduction)

本项目实现了一个“神经网络内部状态实时监测仪”。不同于传统的仅输出分类结果的 Demo，本项目致力于 **Explainable AI (XAI)** 的可视化研究。

系统采用 **双引擎架构**：
1.  **Python 端**：训练一个极简的 NanoMLP (784 → 32 → 10) 并导出为包含中间层状态的 ONNX 模型。
2.  **JavaFX 端**：基于 `onnxruntime` 进行推理，并使用自定义的 `SpectrumCanvas` 组件实时绘制神经网络的 **稀疏激活 (Sparse Activation)** 状态。

## ✨ 核心特性 (Key Features)

* **轻量级模型 (NanoMLP)**：针对实时可视化优化的 3层全连接网络，推理延迟 < 2ms。
* **频谱可视化 (Spectrum Canvas)**：
    * 使用 JavaFX `Canvas API` 替代传统 UI 组件，实现高性能渲染。
    * **Input Layer**: 28x28 原始像素灰度图。
    * **Hidden Layer**: 32维特征向量的“条形码”可视化（亮度映射激活强度）。
    * **Output Layer**: 动态置信度柱状图。
* **跨语言技术闭环**：实现了从 PyTorch (训练) 到 Java (部署) 的完整 MLOps 流程。
* **数据预处理复现**：在 Java 端完整复现了 PyTorch 的 `Normalize((0.1307,), (0.3081,))` 预处理逻辑。

## 🛠 技术栈 (Tech Stack)

* **开发语言**: Java 8 (UI & Inference), Python 3.11 (Model Training)
* **GUI 框架**: JavaFX (JDK 8 内置)
* **推理引擎**: Microsoft ONNX Runtime (Java API)
* **深度学习**: PyTorch, TorchVision
* **构建工具**: Maven

## 📂 项目结构 (Structure)

```text
spectrum-viz/
├── python_model/            # [Python] 模型训练与工具脚本
│   ├── NanoMLP.py      # 训练模型并导出兼容 Java 的 ONNX 文件
│   ├── save_images.py       # 从 MNIST 数据集提取 PNG 测试图片
│   └── requirements.txt
├── src/
│   ├── main/
│   │   ├── java/com/nanomlp/
│   │   │   ├── MainApp.java          # JavaFX 主程序入口
│   │   │   └── ui/SpectrumCanvas.java # 核心可视化绘图组件
│   │   └── resources/
│   │       ├── models/      # 存放 ONNX 模型文件
│   │       └── images/      # 存放 MNIST 测试图片 (.png)
└── pom.xml                  # Maven 依赖配置
