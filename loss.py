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


class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weight, weights):
        super().__init__()
        self.r_weight = weights['reconstruction']
        self.f_weight = weights['forecasting']
        self.c_weight = weights['classification']

        self.r_loss = nn.MSELoss()
        self.f_loss = nn.MSELoss()
        self.c_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, reconstruct, forecast, classification, x_batch, future_batch, y_batch):
        loss =  self.r_weight * self.r_loss(reconstruct, x_batch)
        loss += self.f_weight * self.f_loss(forecast, future_batch)
        loss += self.c_weight * self.c_loss(classification, y_batch)
        return loss
