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

        self.r_criterion = nn.MSELoss()
        self.f_criterion = nn.MSELoss()
        self.c_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, reconstruct, forecast, classification, x_batch, future_batch, y_batch):
        r_loss = self.r_weight * self.r_criterion(reconstruct, x_batch)
        f_loss = self.f_weight * self.f_criterion(forecast, future_batch)
        c_loss = self.c_weight * self.c_criterion(classification, y_batch)
        return r_loss + f_loss + c_loss
