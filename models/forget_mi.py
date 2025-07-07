https://github.com/BioMedIA-MBZUAI/Forget-MI.git
'''
import torch
import torch.nn as nn

class ForgetMI(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original_model = model
        self.unlearned_model = model

    def unlearn(self, forget_loader, noise_level=0.1):
        for batch in forget_loader:
            inputs = batch['image']  # or other modalities
            with torch.no_grad():
                noise = noise_level * torch.randn_like(inputs)
                self.unlearned_model.linear.weight.add_(noise)
        return self.unlearned_model
'''