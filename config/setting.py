"""全局配置：路径、数据集信息与默认超参数。"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "datasets" / "RetinalOCT_Dataset"
OUTPUT_ROOT = ROOT / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"
WEIGHT_DIR = OUTPUT_ROOT / "weights"
COMPARE_DIR = OUTPUT_ROOT / "compare"

for _d in (FIGURE_DIR, LOG_DIR, WEIGHT_DIR, COMPARE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 老师推荐的视网膜 OCT 数据集，8 个类别，已划分好训练/验证/测试集
CLASSES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

IMAGE_SIZE = 128
IN_CHANNELS = 3

SEED = 42
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 基准模型配置，对照实验在此基础上只改动一个变量
BASELINE = {
    "base_channels": 32,
    "kernel_size": 3,
    "depth": 4,
    "dropout": 0.3,
    "pool_size": 2,
    "fc_hidden": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 25,
}

# 实验四：多网络对比的全部网络，统一从头随机初始化训练
MODELS = ["custom", "alexnet", "zfnet", "vgg", "googlenet", "resnet", "mobilenet"]

# 实验四统一训练配置，所有网络共用同一套超参数以保证对比公平
COMPARE = {
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 20,
}

# 上一阶段在旧数据集上通过对照实验筛选出的验证集最优配置
BEST = {"dropout": 0.0}

# 对照实验为控制算力，使用固定的类别等量子集与统一的训练轮次
SEARCH_SUBSETS = {"train": 400, "val": 100}
SEARCH_EPOCHS = 15


def build_args(**overrides):
    """由基准配置生成一次实验的完整参数。"""
    args = dict(BASELINE)
    args.update(overrides)
    return args


def set_seed(seed=SEED):
    """固定随机种子，保证同配置实验可复现。"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)