import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold

    def forward(self, original_embeddings, unlearned_embeddings):
        cos_sim = F.cosine_similarity(original_embeddings, unlearned_embeddings, dim=-1)
        loss = torch.clamp(cos_sim - self.threshold, min=0).mean()
        return loss