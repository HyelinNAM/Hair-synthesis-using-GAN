import torch
from torch import nn
import torchvision

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier,self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        self.base = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,2)
        )

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x