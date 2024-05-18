import torch
from torchvision import transforms
from PIL import Image


def show_tensor_image(tensor: torch.FloatTensor):
    """
    将pytorch张量转化为PIL.Image对象并展示

    Args:
        tensor: pytorch张量，要求shape为(C, H, W)或(B, C, H, W)，其中C表示通道数，H和W表示图像的高和宽

    Returns:
        None
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 如果是batch格式的张量，取第一张图
    tensor = tensor.cpu().clone().detach()  # 将张量移动到CPU上，并克隆一个副本用于可视化
    tensor = tensor.squeeze(0)  # 如果是单张图像的张量，去除batch维度

    un_loader = transforms.ToPILImage()  # 创建一个将tensor转化为PIL.Image对象的转换器
    image: Image = un_loader(tensor)  # 将张量转化为PIL.Image对象

    image.show()  # 展示图像
    pass
