# Sketch2Anime

将素描(Sketch)转换为高质量二次元风格图像的深度学习项目。

## 项目简介

本项目基于Stable Diffusion XL和T2I-Adapter技术，实现从简单素描到精美二次元风格图像的生成。我们提供了两种方法来优化素描到图像的转换过程：

1. **Canny参数优化**：优化Canny边缘检测的超参数，使素描转Canny边缘的过程中尽可能保留细节
2. **模型微调**：直接使用素描作为控制信号来微调T2I-Adapter模型

## 项目结构

```
sketch2anime/
├── configs/             # 配置文件
├── data/                # 数据集存放位置
├── data_processing/     # 数据预处理代码
├── inference/           # 推理代码
├── models/              # 预训练模型保存位置
├── scripts/             # 实用脚本
├── src/                 # 核心代码
└── train/               # 训练代码
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/sketch2anime.git
cd sketch2anime

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

将sketch-image paired数据集放置于`data/`目录下。

### 2. Canny参数优化

运行Canny参数优化实验：

```bash
python scripts/optimize_canny.py --input_dir data/sketches --output_dir data/processed
```

### 3. 模型训练

```bash
python train/train_adapter.py --config configs/train_config.yaml
```

### 4. 推理生成

```bash
python inference/generate.py --sketch path/to/your/sketch.jpg --output output.png
```

## 许可证

MIT License 