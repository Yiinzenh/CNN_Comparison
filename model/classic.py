"""经典卷积神经网络：ZFNet 手动复现，其余调用 torchvision 内置实现。"""
import torch
import torch.nn as nn
from torchvision import models

from config.setting import IN_CHANNELS, NUM_CLASSES


class ZFNet(nn.Module):
    """参考 ZFNet 论文复现：首层卷积改为 7x7 步长 2，并调整中间层通道数。

    相对 AlexNet 的改动：conv1 由 11x11/4 改为 7x7/2，卷积路径为
    96-256-384-384-256，全连接部分保持一致，LRN 已省略以保证可训练性。
    """

    def __init__(self, num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def _alexnet(num_classes):
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def _vgg(num_classes):
    # 16 层无残差网络需 BatchNorm 才能在统一学习率下收敛，故用带 BN 的官方变体
    model = models.vgg16_bn(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def _googlenet(num_classes):
    model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _mobilenet(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


BUILDERS = {
    "alexnet": _alexnet,
    "zfnet": lambda num_classes: ZFNet(num_classes=num_classes),
    "vgg": _vgg,
    "googlenet": _googlenet,
    "resnet": _resnet,
    "mobilenet": _mobilenet,
}

# 各网络的关键设计创新点，用于静态架构对比表
KEY_IDEAS = {
    "custom": "手动堆叠 4 组卷积-激活-池化模块",
    "alexnet": "首个深度 CNN，大卷积核 + 全连接分类头",
    "zfnet": "可视化解剖驱动的调参，首层 7x7/2 卷积",
    "vgg": "3x3 小卷积核重复堆叠，结构规整",
    "googlenet": "Inception 多分支模块 + 全局平均池化",
    "resnet": "残差连接缓解深层网络退化",
    "mobilenet": "深度可分离卷积，轻量化设计",
}


def build(name, num_classes=NUM_CLASSES):
    """按名称构建经典网络，输出层统一改为当前任务的类别数。"""
    if name not in BUILDERS:
        raise KeyError(f"未知网络: {name}")
    return BUILDERS[name](num_classes)