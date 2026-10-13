"""训练循环：逐轮记录训练/验证损失与准确率，并保存最优权重与日志。"""
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from config.setting import DEVICE, LOG_DIR, WEIGHT_DIR


class Trainer:
    def __init__(self, model, args, device=DEVICE, tag="run"):
        self.model = model.to(device)
        self.args = dict(args)
        self.device = device
        self.tag = tag
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"],
                                          weight_decay=args["weight_decay"])
        self.history = None
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_time = 0.0

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs, verbose=True):
        history = {"epoch": [], "train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
        best_state, best_acc, best_epoch = None, 0.0, 0
        start = time.time()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = self._run_epoch(train_loader, True)
            val_loss, val_acc = self._run_epoch(val_loader, False)
            history["epoch_time"].append(round(time.time() - epoch_start, 2))
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(f"  epoch {epoch:>3}/{epochs} | train loss {train_loss:.4f} "
                      f"acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}",
                      flush=True)
        self.train_time = time.time() - start
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history = history
        self.best_val_acc = best_acc
        self.best_epoch = best_epoch
        return history

    def save(self, extra=None):
        """保存最优权重、逐轮日志与超参数，保证实验可复现。"""
        torch.save(self.model.state_dict(), Path(WEIGHT_DIR) / f"{self.tag}.pt")
        record = {
            "tag": self.tag,
            "args": self.args,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "train_time": round(self.train_time, 2),
            "history": self.history,
        }
        if extra:
            record.update(extra)
        (Path(LOG_DIR) / f"{self.tag}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                   "lr", "epoch_time"]
        with open(Path(LOG_DIR) / f"{self.tag}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in zip(*[self.history[k] for k in columns]):
                writer.writerow(row)
        return record


def load_model(model, tag):
    """载入指定实验的权重，用于测试集评估。"""
    state = torch.load(Path(WEIGHT_DIR) / f"{tag}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model