from torch import nn as nn

from mmdp.models.builder import MODULES


@MODULES.register_module()
class CodebookLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self):

        return self.loss_weight
