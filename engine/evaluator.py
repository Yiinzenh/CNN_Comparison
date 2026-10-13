"""模型评估：准确率、混淆矩阵、逐类指标与错分样本收集。"""
import numpy as np
import torch

from config.setting import CLASSES, DEVICE, NUM_CLASSES


@torch.no_grad()
def predict(model, loader, device=DEVICE):
    """按数据顺序返回预测概率、预测标签与真实标签。"""
    model.eval().to(device)
    probs, preds, labels = [], [], []
    for images, targets in loader:
        images = images.to(device)
        batch_prob = torch.softmax(model(images), dim=1)
        probs.append(batch_prob.cpu())
        preds.append(batch_prob.argmax(dim=1).cpu())
        labels.append(targets)
    return (torch.cat(probs).numpy(), torch.cat(preds).numpy(),
            torch.cat(labels).numpy())


def confusion_matrix(preds, labels, num_classes=NUM_CLASSES):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        matrix[true, pred] += 1
    return matrix


def per_class_metrics(matrix):
    """由混淆矩阵计算各类别的精确率、召回率与 F1。"""
    rows = []
    for i in range(len(matrix)):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class": CLASSES[i],
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(matrix[i, :].sum()),
        })
    return rows


def misclassified(records, preds, labels, probs, topk=None):
    """收集错分样本，按预测置信度从高到低排序，便于分析典型错误。"""
    items = []
    for i, (true, pred) in enumerate(zip(labels, preds)):
        if true != pred:
            items.append({
                "path": records[i]["path"],
                "true": CLASSES[true],
                "pred": CLASSES[pred],
                "confidence": round(float(probs[i, pred]), 4),
            })
    items.sort(key=lambda d: -d["confidence"])
    return items[:topk] if topk else items


def evaluate(model, loader, records=None):
    """一次评估返回准确率、混淆矩阵、逐类指标与错分列表。"""
    probs, preds, labels = predict(model, loader)
    accuracy = float((preds == labels).mean())
    matrix = confusion_matrix(preds, labels)
    result = {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": matrix,
        "per_class": per_class_metrics(matrix),
    }
    if records is not None:
        result["misclassified"] = misclassified(records, preds, labels, probs)
    return result