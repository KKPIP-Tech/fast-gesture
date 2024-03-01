import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_pos=100.0, weight_neg=1.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
    
    def forward(self, input, target):
        weights = target * self.weight_pos + (1 - target) * self.weight_neg
        bce_loss = F.binary_cross_entropy_with_logits(input, target, weight=weights)
        return bce_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, input, target):
        smooth = 1.0
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_score


class CombinedLoss(nn.Module):
    def __init__(self, weight_pos=100.0, weight_neg=1.0, lambda_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.weighted_bce_loss = WeightedBCELoss(weight_pos, weight_neg)
        self.dice_loss = DiceLoss()
        self.lambda_dice = lambda_dice
    
    def forward(self, input, target):
        bce_loss = self.weighted_bce_loss(input, target)
        dice_loss = self.dice_loss(torch.sigmoid(input), target)
        combined_loss = bce_loss + self.lambda_dice * dice_loss
        return combined_loss