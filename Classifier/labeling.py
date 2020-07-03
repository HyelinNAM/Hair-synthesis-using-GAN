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

# Gender / Style

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

# Gender
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
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

Gender = Net()
Gender = torch.load('drive/My Drive/BOAZ Adv/최종 model/<최종> GenderClassifier - 96.88') # Gender 모델 path
Gender.to(device)

# Style
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
Hair = torch.load('/home/namhyelin99/adv_data/Model/CBAM classifier - 99 95 90')  # Style 모델 path
Hair.to(device)

# EVAL #
Gender.eval()
Hair.eval()

gender_list = []
color_list = []
bang_list = []
style_list = []

for i, data in tqdm(enumerate(dataloader,0)):

  img, _ = data

  img = img.to(device)
  
  gender = Gender(img)
  color,bang,style = Hair(img)

  _, gender = torch.max(gender,1)
  _, color = torch.max(color,1)
  _, bang = torch.max(bang,1)
  _, style = torch.max(style,1)

  gender_list.append(gender.tolist())
  color_list.append(color.tolist())
  bang_list.append(bang.tolist())
  style_list.append(style.tolist())

  label = {'iter':i,'gender':gender_list,'color':color_list,'bang':bang_list,'style':style_list}

  # 코랩에선 중간에 끊길까봐 iter마다 저장하게 한건데 로컬이면 그냥 다 끝나고 한번만 저장해도 될 듯.
  with open('/home/namhyelin99/label_style.json','w') as f: # json 파일 저장할 루트
    json.dump(label,f,indent=2)

# json > csv
with open("drive/My Drive/label_style.json", 'r') as f: #json 파일 저장된 루트
    label = json.load(f)

gender = sum(label['gender'],[])
color = sum(label['color'],[])
bang = sum(label['bang'],[])
style = sum(label['style'],[])

gender = pd.DataFrame({'gender':gender})
color = pd.DataFrame({'color':color})
bang = pd.DataFrame({'bang':bang})
style = pd.DataFrame({'style':style})

label = pd.concat([gender,bang,color,style],axis=1)
label

label.to_csv('drive/My Drive/celebA_label.csv') # csv 저장할 루트

# 데이터 개수 확인
label['gender'].value_counts()
label['color'].value_counts()
label['bang'].value_counts()
label['style'].value_counts()





