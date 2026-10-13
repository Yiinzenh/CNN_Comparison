"""手动堆叠的自定义卷积神经网络，不使用任何现成经典网络结构。"""
import torch
import torch.nn as nn

from config.setting import IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES


def conv_block(in_channels, out_channels, kernel_size, pool_size, dropout):
    """卷积 - 批归一化 - 激活 - 池化 - Dropout 组成一个特征提取模块。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_size, stride=pool_size),
        nn.Dropout2d(dropout),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=IN_CHANNELS,
                 image_size=IMAGE_SIZE, base_channels=32, kernel_size=3,
                 depth=4, dropout=0.3, pool_size=2, fc_hidden=128):
        super().__init__()
        channels = in_channels
        blocks = []
        for i in range(depth):
            out_channels = min(base_channels * 2 ** i, 512)
            blocks.append(conv_block(channels, out_channels, kernel_size,
                                     pool_size, dropout))
            channels = out_channels
        self.features = nn.Sequential(*blocks)

        # 用一次前向计算特征图展平后的维度，避免手工推算导致维度不匹配
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, image_size, image_size)
            self.flat_dim = int(self.features(sample).numel())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        self.channels = channels

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(args):
    """根据实验配置创建模型，并给出各层特征图维度说明。"""
    model = CustomCNN(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        image_size=IMAGE_SIZE,
        base_channels=args["base_channels"],
        kernel_size=args["kernel_size"],
        depth=args["depth"],
        dropout=args["dropout"],
        pool_size=args["pool_size"],
        fc_hidden=args["fc_hidden"],
    )
    return model


def describe(args, model=None):
    """输出每层输出特征图维度，便于核对维度流转是否正确。"""
    size = IMAGE_SIZE
    lines = [f"输入: {IN_CHANNELS} x {size} x {size}"]
    for i in range(args["depth"]):
        out_channels = min(args["base_channels"] * 2 ** i, 512)
        size = size // args["pool_size"]
        lines.append(f"模块{i + 1}: Conv{args['kernel_size']}x{args['kernel_size']} "
                     f"-> {out_channels} x {size} x {size}")
    model = model or build_model(args)
    lines.append(f"展平后维度: {model.flat_dim}")
    lines.append(f"全连接层: {model.flat_dim} -> {args['fc_hidden']} -> {NUM_CLASSES}")
    return lines


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)