"""图像预处理：统一尺寸缩放与像素归一化。"""
from torchvision import transforms

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def build_transform(image_size, train=False):
    """训练与验证使用同一套确定性的缩放和归一化，保证对照实验可复现。"""
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)