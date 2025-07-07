import torch.nn as nn
import torchvision.models as vision_models
from transformers import BertModel

class BaseResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = vision_models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


class BaseBERT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return self.classifier(outputs.last_hidden_state.mean(dim=1))


class BaseLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])