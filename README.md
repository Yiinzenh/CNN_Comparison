# CNN_Comparison

基于 PyTorch 的图像分类代码库，包含一个手动堆叠的自定义卷积神经网络，以及多个经典卷积网络的统一训练、评估与横向对比流程。

## 环境依赖

- Python 3.x
- PyTorch、torchvision
- numpy、pandas、matplotlib、Pillow

有 GPU 时会自动使用 CUDA，否则回落到 CPU（见 `config/setting.py` 中的 `DEVICE`）。

## 数据集准备

数据集目录需按「划分 / 类别」两级组织，类别名与 `config/setting.py` 中的 `CLASSES` 一致：

```
datasets/RetinalOCT_Dataset/
├── train/
│   ├── AMD/    *.jpg
│   ├── CNV/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

若使用其他数据集，修改 `config/setting.py` 中的 `DATA_ROOT`、`CLASSES`、`NUM_CLASSES`、`IMAGE_SIZE`、`IN_CHANNELS` 即可。

## 目录结构

```
main.py                 命令行入口
config/setting.py       路径、数据集信息与超参数配置
data/dataset.py         数据集扫描、统计与 DataLoader 构建
data/transform.py       图像预处理与数据增强
model/cnn.py            自定义 CNN（手动堆叠卷积-激活-池化模块）
model/classic.py        经典网络定义（AlexNet、ZFNet、VGG、GoogLeNet、ResNet、MobileNetV3）
engine/trainer.py       训练循环、日志与权重保存
engine/evaluator.py     混淆矩阵、逐类指标与预测
engine/cross.py         多网络错分样本交叉分析
experiments/analyze.py  静态架构分析（参数量、FLOPs）
experiments/search.py   控制变量法超参数对照实验
experiments/compare.py  多网络统一训练与评估
utils/flops.py          参数量与 FLOPs 统计
utils/plot.py           曲线、混淆矩阵与对比图绘制
```

## 使用方法

所有功能通过 `python main.py <子命令>` 调用。

### 1. 数据集统计

统计样本总量、类别分布、图像尺寸与通道数，保存类别分布图与样本示例图。

```bash
python main.py stats
```

### 2. 静态架构分析

打印各网络完整结构，统计卷积层、池化层、归一化层、Dropout 层数量与参数量、FLOPs，生成架构对比表与柱状图。不执行训练。

```bash
python main.py analyze
python main.py analyze --names custom alexnet resnet
```

### 3. 基础模型训练

按 `config/setting.py` 中 `BEST` 覆盖后的配置训练自定义 CNN。

```bash
python main.py train
python main.py train --tag custom
```

### 4. 控制变量法对照实验

每个实验组只改动一个超参数，其余保持不变。可用组名：`kernel`、`depth`、`dropout`、`lr`、`pool`、`epoch`。

```bash
python main.py search
python main.py search --groups lr depth
```

### 5. 多网络统一训练

在统一超参数下训练指定网络。支持逐个训练，每训练完一个网络即增量更新汇总表并重绘对比曲线，中途中断不会丢失已完成结果。

```bash
python main.py compare
python main.py compare --names resnet
python main.py compare --names custom vgg --epochs 25
```

可用网络名：`custom`、`alexnet`、`zfnet`、`vgg`、`googlenet`、`resnet`、`mobilenet`。

### 6. 统一评估与交叉分析

对权重已存在的网络做统一评估，输出混淆矩阵、逐类指标与错分样本交叉分析。

```bash
python main.py report
python main.py report --split test
python main.py report --names custom resnet --split test
```

`--split` 可选 `val`（默认）或 `test`。

### 7. 单模型测试集评估

对指定的单个模型在测试集上评估，输出准确率、逐类指标、混淆矩阵与错分样本图。

```bash
python main.py eval --tag resnet
```

## 输出目录

| 目录 | 内容 |
| --- | --- |
| `outputs/weights/` | 各网络权重（`*.pt`） |
| `outputs/logs/` | 逐轮训练日志（`*.json` / `*.csv`）、汇总表、评估明细 |
| `outputs/compare/` | 多网络对比曲线、混淆矩阵、架构对比图 |
| `outputs/figures/` | 类别分布图、样本示例图、单模型训练曲线 |

其中 `datasets/` 与 `outputs/` 已在 `.gitignore` 中排除，不纳入版本管理。

## 配置说明

`config/setting.py` 集中管理全部可调项：

| 配置项 | 说明 |
| --- | --- |
| `BASELINE` | 自定义 CNN 的基准超参数（通道数、卷积核、深度、Dropout、池化窗口、批大小、学习率、轮数等） |
| `BEST` | 覆盖基准配置的最终模型配置 |
| `COMPARE` | 多网络统一训练配置 |
| `SEARCH_SUBSETS` / `SEARCH_EPOCHS` | 对照实验使用的数据子集规模与训练轮次 |
| `SEED` | 随机种子，用于结果可复现 |
