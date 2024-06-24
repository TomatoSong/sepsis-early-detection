import torch
import torch.nn as nn

class UtilityLoss(nn.Module):
    def __init__(self):
        super(UtilityLoss, self).__init__()

    def forward(self, outputs, targets, u_weights):
        loss = -torch.sum((1 - outputs) * u_weights[:, 0] + outputs * u_weights[:, 1])
        return loss
