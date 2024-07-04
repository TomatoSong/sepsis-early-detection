import torch
import torch.nn as nn

class UtilityLoss(nn.Module):
    def __init__(self, pos_weight=1):
        super(UtilityLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, outputs, targets, u_weights):
        loss = -((1 - outputs) * u_weights[:, 0] + outputs * u_weights[:, 1])
        loss += self.pos_weight * targets * loss
        return torch.sum(loss) / len(outputs)
