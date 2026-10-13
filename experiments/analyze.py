"""实验四任务2：多网络静态架构分析（不执行训练）。"""
import json
from pathlib import Path

import pandas as pd
import torch.nn as nn

from config.setting import (BEST, COMPARE_DIR, LOG_DIR, MODELS, NUM_CLASSES,
                            build_args)
from model import classic
from model.cnn import build_model
from utils.flops import count_flops, count_params
from utils.plot import plot_metric_bar


def build_all(names=None):
    """构建手搭 CNN 与全部经典网络，权重均为随机初始化。"""
    names = names or MODELS
    return {name: (build_model(build_args(**BEST)) if name == "custom"
                   else classic.build(name, NUM_CLASSES)) for name in names}


def describe(name, model):
    """统计单个网络的层数、参数量与计算量。"""
    convs = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    pools = sum(1 for m in model.modules()
                if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)))
    norms = sum(1 for m in model.modules()
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)))
    drops = sum(1 for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d)))
    total, trainable = count_params(model)
    return {
        "network": name,
        "conv": convs,
        "pool": pools,
        "norm": norms,
        "dropout": drops,
        "params": total,
        "trainable": trainable,
        "flops": count_flops(model),
        "key_idea": classic.KEY_IDEAS[name],
    }


def analyze(names=None):
    """打印每个网络的结构与指标，保存架构对比表与柱状图。"""
    rows = []
    for name, model in build_all(names).items():
        print("=" * 70)
        print(f"[{name}] {classic.KEY_IDEAS[name]}")
        print(model)
        row = describe(name, model)
        rows.append(row)
        print(f"参数量 {row['params']:,} | 可训练 {row['trainable']:,} | "
              f"FLOPs {row['flops'] / 1e9:.2f}G")

    table = pd.DataFrame(rows)
    table.to_csv(Path(LOG_DIR) / "arch.csv", index=False, encoding="utf-8-sig")
    (Path(LOG_DIR) / "arch.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    names_list = table["network"].tolist()
    plot_metric_bar(names_list, table["params"].tolist(),
                    "Parameter count of each network", "arch_params.png",
                    folder=COMPARE_DIR, ylabel="Params (M)", scale=1e6)
    plot_metric_bar(names_list, table["flops"].tolist(),
                    "Computational cost of each network", "arch_flops.png",
                    folder=COMPARE_DIR, ylabel="FLOPs (G)", scale=1e9)
    print(table.to_string(index=False))
    return table