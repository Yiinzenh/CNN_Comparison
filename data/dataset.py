"""数据集扫描、划分信息统计与 DataLoader 构建。"""
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config.setting import (CLASSES, CLASS_TO_IDX, DATA_ROOT,
                            IMAGE_EXTENSIONS, IMAGE_SIZE, IN_CHANNELS, LOG_DIR,
                            NUM_WORKERS, SEED, SPLITS)


def scan_split(split, root=DATA_ROOT):
    """扫描某个划分目录，类别由子目录名确定。"""
    records = []
    for cls in CLASSES:
        folder = Path(root) / split / cls
        for path in sorted(p for p in folder.iterdir()
                           if p.suffix.lower() in IMAGE_EXTENSIONS):
            records.append({"path": str(path), "label": CLASS_TO_IDX[cls],
                            "name": cls, "split": split})
    return records


def scan(root=DATA_ROOT):
    """数据集已划分好 train/val/test，直接读取三个划分目录。"""
    return {split: scan_split(split, root) for split in SPLITS}


class OCTDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = Image.open(rec["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, rec["label"]


def build_loaders(records, train_transform, eval_transform, batch_size):
    """训练集使用带增广的变换，验证与测试集使用确定性变换并保持顺序。"""
    kwargs = {"num_workers": NUM_WORKERS, "persistent_workers": NUM_WORKERS > 0}
    loaders = {}
    for split in records:
        transform = train_transform if split == "train" else eval_transform
        loaders[split] = DataLoader(OCTDataset(records[split], transform),
                                    batch_size=batch_size,
                                    shuffle=(split == "train"), **kwargs)
    return loaders


def subsample(records, per_class, seed=SEED):
    """按类别等量抽取固定子集，保证各组对照实验使用完全相同的样本。"""
    rng = random.Random(seed)
    kept = []
    for name in CLASSES:
        pool = [r for r in records if r["name"] == name]
        kept.extend(rng.sample(pool, min(per_class, len(pool))))
    return kept


def prepare_loaders(batch_size, image_size=IMAGE_SIZE, subsets=None, splits=None):
    """读取数据集并提供统一的数据加载器，供训练与评估复用。

    subsets 给出各划分每类抽取的样本数，仅用于算力受限的对照实验；
    splits 用于只构建所需划分，对照实验阶段不加载测试集。
    """
    from data.transform import build_transform

    records = scan()
    if subsets:
        records = {s: (subsample(records[s], subsets[s]) if s in subsets else records[s])
                   for s in SPLITS}
    if splits:
        records = {s: records[s] for s in splits}
    loaders = build_loaders(records, build_transform(image_size, train=True),
                            build_transform(image_size, train=False), batch_size)
    return records, loaders


def summarize(records):
    """统计样本总量、类别数量、图像分辨率与通道数。"""
    counts = {split: defaultdict(int) for split in SPLITS}
    widths, heights = set(), set()
    for split in SPLITS:
        for rec in records[split]:
            counts[split][rec["name"]] += 1
            with Image.open(rec["path"]) as image:
                widths.add(image.size[0])
                heights.add(image.size[1])
    info = {
        "total": sum(len(records[s]) for s in SPLITS),
        "classes": CLASSES,
        "channels": IN_CHANNELS,
        "image_size": IMAGE_SIZE,
        "split_counts": {s: len(records[s]) for s in SPLITS},
        "class_counts": {s: {c: counts[s].get(c, 0) for c in CLASSES} for s in SPLITS},
        "raw_width_range": [min(widths), max(widths)],
        "raw_height_range": [min(heights), max(heights)],
    }
    return info


def save_summary(info, path=LOG_DIR / "dataset.json"):
    """保存数据集统计与划分信息，保证实验记录完整。"""
    Path(path).write_text(json.dumps(info, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return path