"""实验四任务3/4：统一条件下多网络训练与评估。

支持逐个训练：compare --names X 只训练指定网络并增量更新汇总表；
全部训练完成后用 report 统一评估并做错分样本交叉分析。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.setting import (BEST, COMPARE, COMPARE_DIR, DEVICE, LOG_DIR, MODELS,
                            NUM_CLASSES, SEED, WEIGHT_DIR, build_args, set_seed)
from data.dataset import prepare_loaders
from engine.cross import cross_analysis
from engine.evaluator import confusion_matrix, per_class_metrics, predict
from engine.trainer import Trainer, load_model
from model import classic
from model.cnn import build_model
from utils.flops import count_params
from utils.plot import plot_compare_curves, plot_confusion, plot_curves


def build_network(name):
    """按名称构建网络，手搭 CNN 沿用实验三筛选出的最优配置。"""
    if name == "custom":
        return build_model(build_args(**BEST))
    return classic.build(name, NUM_CLASSES)


def trained_names():
    """返回权重目录中已训练完成的网络，保持统一顺序。"""
    return [name for name in MODELS if (Path(WEIGHT_DIR) / f"{name}.pt").exists()]


def load_histories(names):
    """读取已保存的训练日志，用于绘制多网络对比曲线。"""
    histories = {}
    for name in names:
        path = Path(LOG_DIR) / f"{name}.json"
        if path.exists():
            histories[name] = json.loads(path.read_text(encoding="utf-8"))["history"]
    return histories


def update_table(summaries):
    """把本次训练结果合并进汇总表，支持逐网络增量训练。"""
    path = Path(LOG_DIR) / "compare.csv"
    new = pd.DataFrame(summaries)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["network"].isin(new["network"])]
        new = pd.concat([old, new], ignore_index=True)
    order = [n for n in MODELS if n in set(new["network"])]
    new = new.set_index("network").loc[order].reset_index()
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return new


def train(names, epochs, loaders, args):
    """依次从头训练指定网络，逐轮记录指标并更新汇总表与对比曲线。"""
    summaries = []
    for name in names:
        set_seed(SEED)
        model = build_network(name)
        total, _ = count_params(model)
        trainer = Trainer(model, args, device=DEVICE, tag=name)
        trainer.fit(loaders["train"], loaders["val"], epochs)
        trainer.save(extra={"group": "compare", "params": total})
        plot_curves(trainer.history, f"{name} (best val acc {trainer.best_val_acc:.4f})",
                    f"curves_{name}.png", folder=COMPARE_DIR)
        summaries.append({
            "network": name,
            "params": total,
            "train_acc": round(trainer.history["train_acc"][-1], 4),
            "val_acc": round(trainer.best_val_acc, 4),
            "best_epoch": trainer.best_epoch,
            "epoch_time": round(float(np.mean(trainer.history["epoch_time"])), 2),
            "time": round(trainer.train_time, 1),
        })
        table = update_table(summaries)
        print(f"\n[{name}] 训练完成，当前汇总表:\n{table.to_string(index=False)}", flush=True)

    plot_compare_curves(load_histories(trained_names()),
                        "Validation loss and accuracy of all networks",
                        "compare_curves.png", folder=COMPARE_DIR)
    return summaries


def report(names=None, split="val"):
    """在指定数据划分（val/test）上评估已训练网络，输出混淆矩阵与错分交叉分析。"""
    names = names or trained_names()
    records, loaders = prepare_loaders(COMPARE["batch_size"])
    suffix = "" if split == "val" else f"_{split}"
    preds_by_net, blocks, accs = {}, [], {}
    for name in names:
        model = build_network(name)
        load_model(model, name)
        _, preds, labels = predict(model, loaders[split])
        accs[name] = round(float((preds == labels).mean()), 4)
        matrix = confusion_matrix(preds, labels)
        plot_confusion(matrix, f"Confusion matrix ({name} on {split})",
                       f"confusion{suffix}_{name}.png", folder=COMPARE_DIR)
        block = pd.DataFrame(per_class_metrics(matrix))
        block.insert(0, "network", name)
        blocks.append(block)
        preds_by_net[name] = preds
    pd.concat(blocks).to_csv(Path(LOG_DIR) / f"compare_class{suffix}.csv",
                             index=False, encoding="utf-8-sig")

    cross = cross_analysis(preds_by_net, labels, records[split]) if len(names) > 1 else None
    if cross:
        (Path(LOG_DIR) / f"compare_cross{suffix}.json").write_text(
            json.dumps(cross, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n被全部网络错分的样本数: {len(cross['hard_samples'])}")
        print(f"手搭 CNN 独有错误样本数: {len(cross['custom_gap_samples'])}")
    print(f"{split} 集准确率: {accs}")
    return accs, cross


def compare(names=None, epochs=None):
    """一次性完成指定网络的训练与评估。"""
    names = names or MODELS
    args = dict(COMPARE, epochs=epochs or COMPARE["epochs"])
    records, loaders = prepare_loaders(args["batch_size"])
    train(names, args["epochs"], loaders, args)
    return report(names)