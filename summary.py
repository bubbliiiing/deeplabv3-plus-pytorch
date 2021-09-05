#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=21, backbone="mobilenet", downsample_factor=16, pretrained=False).to(device)
    summary(model, (3,512,512))
