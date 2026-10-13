"""绘图工具：训练曲线、混淆矩阵、类别分布、样本网格与对照实验柱状图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config.setting import CLASSES, FIGURE_DIR

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TRAIN_COLOR = "#2f6fbb"
VAL_COLOR = "#d1495b"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_curves(history, title, filename, folder=FIGURE_DIR):
    """绘制训练/验证损失与准确率曲线，用于判断过拟合与欠拟合。"""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history["train_loss"], color=TRAIN_COLOR, lw=2, label="Train")
    axes[0].plot(epochs, history["val_loss"], color=VAL_COLOR, lw=2, label="Val")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], color=TRAIN_COLOR, lw=2, label="Train")
    axes[1].plot(epochs, history["val_acc"], color=VAL_COLOR, lw=2, label="Val")
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    fig.suptitle(title)
    return _save(fig, Path(folder) / filename)


def plot_confusion(matrix, title, filename, folder=FIGURE_DIR):
    """绘制混淆矩阵热力图，同时标出各类别样本数。"""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "#333333",
                        fontsize=9)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _save(fig, Path(folder) / filename)


def plot_class_distribution(class_counts, title, filename, folder=FIGURE_DIR):
    """绘制各划分下的类别分布柱状图。"""
    splits = list(class_counts)
    names = [c for c in CLASSES if any(c in class_counts[s] for s in splits)]
    x = np.arange(len(names))
    width = 0.8 / len(splits)

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 4, 4.4))
    for i, split in enumerate(splits):
        values = [class_counts[split].get(c, 0) for c in names]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=split)
        ax.bar_label(bars, fmt="%d", fontsize=7.5, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.set_ylabel("Number of images")
    ax.legend()
    ax.margins(y=0.15)
    return _save(fig, Path(folder) / filename)


def plot_sample_grid(paths, labels, title, filename, cols=4, rows=2, folder=FIGURE_DIR):
    """展示部分样本，确认输入图像内容、尺寸与通道数。"""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 1.7))
    for ax, path, label in zip(axes.ravel(), paths, labels):
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{label}  {image.size[0]}x{image.size[1]}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(paths):]:
        ax.axis("off")
    fig.suptitle(title)
    return _save(fig, Path(folder) / filename)


def plot_misclassified(items, title, filename, cols=4, rows=2, limit=8, folder=FIGURE_DIR):
    """按预测置信度展示典型错分样本。"""
    items = items[:limit]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.0))
    for ax, item in zip(axes.ravel(), items):
        ax.imshow(Image.open(item["path"]).convert("RGB"))
        ax.set_title(f"true {item['true']} | pred {item['pred']}\n"
                     f"conf {item['confidence']:.3f}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title)
    return _save(fig, Path(folder) / filename)


def plot_comparison(results, title, filename, folder=FIGURE_DIR):
    """绘制对照实验的精度对比柱状图，同时显示训练集精度以反映过拟合程度。"""
    values = [str(r["value"]) for r in results]
    train_acc = [r["train_acc"] for r in results]
    val_acc = [r["val_acc"] for r in results]
    x = np.arange(len(values))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.7 * len(values) + 3, 4.6))
    bars_train = ax.bar(x - width / 2, train_acc, width, label="Train acc",
                        color=TRAIN_COLOR)
    bars_val = ax.bar(x + width / 2, val_acc, width, label="Val acc",
                      color=VAL_COLOR)
    ax.bar_label(bars_train, fmt="%.3f", fontsize=8.5)
    ax.bar_label(bars_val, fmt="%.3f", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(values)
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    return _save(fig, Path(folder) / filename)


def plot_compare_curves(histories, title, filename, folder=FIGURE_DIR):
    """把多个网络的验证损失与准确率曲线画在同一张图上，直观对比收敛速度。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for color, (name, history) in zip(colors, histories.items()):
        axes[0].plot(history["epoch"], history["val_loss"], color=color, lw=1.8,
                     label=name)
        axes[1].plot(history["epoch"], history["val_acc"], color=color, lw=1.8,
                     label=name)

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=9)

    fig.suptitle(title)
    return _save(fig, Path(folder) / filename)


def plot_metric_bar(names, values, title, filename, folder=FIGURE_DIR,
                    ylabel="", scale=1.0, fmt="%.2f"):
    """绘制单个指标（参数量、计算量等）的网络对比柱状图。"""
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 3, 4.6))
    bars = ax.bar(np.arange(len(names)), [v / scale for v in values],
                  color=TRAIN_COLOR)
    ax.bar_label(bars, labels=[fmt % (v / scale) for v in values], fontsize=9,
                 padding=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.15)
    return _save(fig, Path(folder) / filename)