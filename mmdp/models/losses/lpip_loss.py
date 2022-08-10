import os
import sys

from torch import nn as nn

from mmdp.models.architectures import LPIPS
from mmdp.models.builder import MODULES

# python程序在命令行执行提示ModuleNotFoundError: No module named 'XXX' 解决方法
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LPIPS_VGG_WEIGHT_PATH = "./work_dirs/Quantexsr/lpips/weights/v0.1/vgg.pth"


@MODULES.register_module()
class LPIPSLoss(nn.Module):
    """LPIPS loss with vgg backbone."""

    def __init__(self, loss_weight=1.0):
        super(LPIPSLoss, self).__init__()
        self.model = LPIPS(net="vgg", pretrained_model_path=LPIPS_VGG_WEIGHT_PATH)
        self.model.eval()
        self.loss_weight = loss_weight

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, gt):
        return self.model(x, gt) * self.loss_weight, None
