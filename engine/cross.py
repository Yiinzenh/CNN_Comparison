"""多网络错分样本集合交叉分析。"""
import numpy as np


def cross_analysis(preds, labels, records):
    """preds 为 {网络名: 预测数组}，records 与预测顺序一致。

    返回被全部网络错分的样本、经典网络正确而手搭 CNN 错误的样本，
    以及每个样本被错分的网络数量，用于分析不同网络的短板。
    """
    names = list(preds)
    labels = np.asarray(labels)
    flags = np.stack([np.asarray(preds[name]) != labels for name in names])
    wrong_count = flags.sum(axis=0)

    def view(indices):
        return [{"path": records[i]["path"], "name": records[i]["name"],
                 "wrong_by": [names[k] for k in range(len(names)) if flags[k, i]]}
                for i in indices]

    hard = [i for i in range(len(labels)) if wrong_count[i] == len(names)]

    reference = [name for name in names if name != "custom"]
    gap = []
    if "custom" in names and reference:
        ref_ok = np.all(~flags[[names.index(n) for n in reference]], axis=0)
        own_wrong = flags[names.index("custom")]
        gap = [i for i in range(len(labels)) if ref_ok[i] and own_wrong[i]]

    return {
        "networks": names,
        "wrong_count": wrong_count.tolist(),
        "hard_samples": view(hard),
        "custom_gap_samples": view(gap),
    }