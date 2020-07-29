import torch
from torch import nn
from CBAM_PyTorch.model.resnet_cbam import resnet34_cbam

class StyleClassifier(nn.Module):
    def __init__(self):
        super(StyleClassifier,self).__init__()
        resnet = resnet34_cbam(pretrained=False)
        self.base = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048,1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512,3)
        ) # Color

        self.fc2 = nn.Sequential(
            nn.Linear(2048,1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512,2)
        ) # Bang

        self.fc3 = nn.Sequential(
            nn.Linear(2048,1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512,5)
        ) # Style

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)

        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)

        return x1,x2,x3