import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareReweightingLoss(nn.Module):
    def __init__(self, base_loss=nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, student_logits, teacher_logits, labels):
        loss_per_sample = self.base_loss(student_logits, labels)
        uncertainties = -(F.softmax(teacher_logits, dim=1) * F.log_softmax(teacher_logits, dim=1)).sum(dim=1)
        weights = 1 + uncertainties.detach()
        return (loss_per_sample * weights).mean()


class FocalLossWithAdaptiveFocusing(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        pt = torch.softmax(logits, dim=-1)
        log_pt = torch.log_softmax(logits, dim=-1)
        focal_weights = (1 - pt) ** self.gamma
        loss = -targets * log_pt * focal_weights
        return loss.mean()


class ContrastiveMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        distances = torch.cdist(embeddings, embeddings)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        neg_mask = ~pos_mask
        positive_distances = distances[pos_mask].view(len(embeddings), -1).mean(dim=1)
        negative_distances = distances[neg_mask].view(len(embeddings), -1).mean(dim=1)
        losses = torch.relu(self.margin + positive_distances - negative_distances)
        return losses.mean()