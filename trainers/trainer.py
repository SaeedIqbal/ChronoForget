import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

class ChronoForgetTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=1e-4)

    def _compute_forget_set_loss(self, inputs, labels, teacher_outputs):
        raise NotImplementedError

    def _compute_retain_set_loss(self, inputs, teacher_inputs):
        raise NotImplementedError

    def train_unlearn(self, forget_loader: DataLoader, retain_loader: DataLoader, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(forget_loader, desc=f"Forgetting Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                loss = self._compute_forget_set_loss(inputs, labels, self.model(inputs))
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"Forget Loss": loss.item()})

            pbar = tqdm(retain_loader, desc=f"Retaining Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                loss = self._compute_retain_set_loss(inputs, labels)
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"Retain Loss": loss.item()})