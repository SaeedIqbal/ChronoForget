import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherStudentDistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=1)
        log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean') * (self.temperature ** 2)
        return loss


class EmbeddingDriftLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original_embeddings, unlearned_embeddings):
        return F.pairwise_distance(original_embeddings, unlearned_embeddings).mean()