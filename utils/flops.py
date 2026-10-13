"""参数量与计算量统计：通过前向钩子累加卷积与全连接的乘加次数。"""
import torch
import torch.nn as nn

from config.setting import IMAGE_SIZE, IN_CHANNELS


def count_params(model):
    """返回总参数量与可训练参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, image_size=IMAGE_SIZE, in_channels=IN_CHANNELS):
    """统计指定输入尺寸下卷积与全连接层的乘加次数（MACs），作为计算量指标。"""
    model = model.eval()
    macs = [0]
    handles = []

    def conv_hook(module, inputs, output):
        kernel = (module.kernel_size[0] * module.kernel_size[1]
                  * (module.in_channels // module.groups))
        macs[0] += output.shape[2] * output.shape[3] * module.out_channels * kernel

    def linear_hook(module, inputs, output):
        macs[0] += module.in_features * module.out_features

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(torch.zeros(1, in_channels, image_size, image_size))
    for handle in handles:
        handle.remove()
    return macs[0]