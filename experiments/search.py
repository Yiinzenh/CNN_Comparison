"""控制变量法对照实验：每次仅改变一个超参数，其余条件保持一致。"""
import json
from pathlib import Path

import pandas as pd

from config.setting import (BASELINE, DEVICE, LOG_DIR, SEARCH_EPOCHS,
                            SEARCH_SUBSETS, SEED, set_seed)
from data.dataset import prepare_loaders
from engine.trainer import Trainer
from model.cnn import build_model, count_params
from utils.plot import plot_comparison, plot_curves

# 每个实验组只修改一个变量；values 中的基准值即为基准模型的取值
GROUPS = {
    "kernel": {"key": "kernel_size", "values": [3, 5, 7],
               "title": "Effect of convolution kernel size"},
    "depth": {"key": "depth", "values": [3, 4, 5, 6],
              "title": "Effect of network depth"},
    "dropout": {"key": "dropout", "values": [0.0, 0.3, 0.5, 0.7],
                "title": "Effect of dropout rate"},
    "lr": {"key": "lr", "values": [1e-2, 1e-3, 1e-4],
           "title": "Effect of learning rate"},
    "pool": {"key": "pool_size", "values": [2, 3],
             "title": "Effect of pooling window size"},
    "epoch": {"key": "epochs", "values": [5, 10, 15],
              "title": "Effect of training epochs"},
}


def run_group(group, spec, loaders, base_args):
    """在固定训练/验证集上逐一评估该超参数的不同取值。"""
    results = []
    for value in spec["values"]:
        args = dict(base_args)
        args[spec["key"]] = value
        tag = f"{group}_{value}"
        print(f"[{tag}] {spec['key']}={value}", flush=True)
        set_seed(SEED)
        model = build_model(args)
        trainer = Trainer(model, args, device=DEVICE, tag=tag)
        trainer.fit(loaders["train"], loaders["val"], args["epochs"])
        trainer.save(extra={"group": group, "value": value,
                            "params": count_params(model)})
        plot_curves(trainer.history, f"{tag}  ({spec['key']}={value})",
                    f"curves_{tag}.png")
        results.append({
            "group": group,
            "value": value,
            "params": count_params(model),
            "train_acc": round(trainer.history["train_acc"][-1], 4),
            "val_acc": round(trainer.best_val_acc, 4),
            "gap": round(trainer.history["train_acc"][-1] - trainer.best_val_acc, 4),
            "best_epoch": trainer.best_epoch,
            "time": round(trainer.train_time, 1),
        })
    plot_comparison(results, spec["title"], f"compare_{group}.png")
    return results


def search(groups=None):
    """依次执行所有对照实验，汇总为表格并绘制对比图。"""
    groups = groups or GROUPS
    loaders = prepare_loaders(BASELINE["batch_size"], subsets=SEARCH_SUBSETS,
                              splits=["train", "val"])[1]
    base_args = dict(BASELINE, epochs=SEARCH_EPOCHS)
    all_results = []
    for group, spec in groups.items():
        all_results.extend(run_group(group, spec, loaders, base_args))

    table = pd.DataFrame(all_results)
    table.to_csv(Path(LOG_DIR) / "search.csv", index=False, encoding="utf-8-sig")
    (Path(LOG_DIR) / "search.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    return table


def best(table):
    """选出验证集精度最高的模型，供测试集最终评估使用。"""
    row = table.sort_values(["val_acc", "gap"], ascending=[False, True]).iloc[0]
    return f"{row['group']}_{row['value']}"