# 임시 라벨링 (모델 implement) 코드

import os
import tqdm
import random
import pandas as pd
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset,random_split,Subset,ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from resnet_cbam import resnet34_cbam

# Style

path = '/home/namhyelin99/googledrive/BOAZ Adv/Data/Face' # 이미지 데이터 path
# '/home/namhyelin99/adv_data/Face'

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = dset.ImageFolder(root=path,transform=transform)
dataloader = DataLoader(image,batch_size=120,shuffle=False)

print(len(dataloader))

class MyResNet(nn.Module):
  def __init__(self):
    super(MyResNet,self).__init__()
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
    )

    self.fc2 = nn.Sequential(
        nn.Linear(2048,1024),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(512,2)
    )

    self.fc3 = nn.Sequential(
        nn.Linear(2048,1024),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(512,5)
    )

  def forward(self,x):
    x = self.base(x)
    x = x.view(x.size(0),-1)

    x1 = self.fc1(x)
    x2 = self.fc2(x)
    x3 = self.fc3(x)

    return x1,x2,x3

Hair = MyResNet()
model_path = '/home/namhyelin99/adv_data/Model/CBAM classifier - 99 95 90'
Hair = torch.load(model_path,map_location='cpu')
#Hair.to(device)

# EVAL #
Hair.eval()

color_list = []
bang_list = []
style_list = []

for i, data in enumerate(dataloader,0):

  img, _ = data

  img = img.to(device)
  
  color,bang,style = Hair(img)

  _, color = torch.max(color,1)
  _, bang = torch.max(bang,1)
  _, style = torch.max(style,1)

  color_list.append(color.tolist())
  bang_list.append(bang.tolist())
  style_list.append(style.tolist())

  label = {'iter':i,'gender':gender_list,'color':color_list,'bang':bang_list,'style':style_list}

  with open('/home/namhyelin99/label_style.json','w') as f:
    json.dump(label,f,indent=2)

  print(i)

torch.cuda.is_available()





