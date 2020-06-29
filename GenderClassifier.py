# Gender Classifier with ResNet
import os
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset,random_split,Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Gender_utils import CustomDataset,early_stopping_and_save_model
from Gender_models import GenderClassifier

class GenderClassifier(self):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        data_F_path = 'drive/My Drive/Adv_개인/Adv_공유/Data/Face' # Face
        label_F_path = 'drive/My Drive/BOAZ Adv_GAN/Adv_공유/Data/face.csv' # 라벨링한 데이터와 path 같음. 수정해야
        data_C_path = 'drive/My Drive/BOAZ Adv/Data/CelebA-HQ'# CelebA-Hq
        label_C_path = 'drive/My Drive/BOAZ Adv_GAN/Adv_공유/Data/CelebA.csv' # 추가해야

        self.path = {'data_F_path':data_F_path,'label_F_path':label_F_path,'data_C_path':data_C_path, 'label_C_path':label_C_path}

    def __call__(self):
        print("Making Custom dataset...")
        self.Face = self.mkdataset()
        train_sampler, valid_sampler, test_sampler = self.datasplit(valid_ratio=0.15,test_ratio=0.15)

        train = DataLoader(self.Face,sampler=train_sampler,batch_size=16,drop_last=True)
        valid = DataLoader(self.Face,sampler=valid_sampler,batch_size=16,drop_last=True)
        test = DataLoader(self.Face,sampler=test_sampler,batch_size=16,drop_last=True)

        print(f'Number of Dataset: train-{len(train)} valid-{len(valid)} test-{len(test)}')

        self.train(train,valid,test)
        
    def mkdataset(self):
        transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
        ])

        transform2 = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
        ])

        Face_F = CustomDataset(self.path['data_F_path'],self.path['label_F_path'],transform)
        Face_F_aug = CustomDataset(self.path['data_F_path'],self.path['label_F_path'],transform2)
        Face_C = CustomDataset(self.path['data_C_path'],self.path['label_C_path'],transform)
        Face_C_aug = CustomDataset(self.path['data_C_path'],self.path['label_C_path'],transform2)
        
        Face = ConcatDataset([Face_F,Face_F_aug,Face_C,Face_C_aug])

        return Face

    def datasplit(valid_ratio,test_ratio):
        np.random.seed(777)

        size = len(self.Face)
        indices = list(range(size))

        np.random.shuffle(indices)

        split_valid = int(np.floor(size*valid_ratio))
        split_test = split_valid+int(np.floor(size*test_ratio))

        train_indices, val_indices, test_indices = indices[split_test:],indices[:split_valid], indices[split_valid:split_test]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        return train_sampler,valid_sampler,test_sampler

    def train(train,valid,test,epochs=200):

        # For ealry stopping
        torch_model_name = "GenderClassifier"
        early_stopping_patience = 15

        print("Training Start...!")

        running_loss_list = []
        valid_loss_list = []
        early_stopping_val_loss_list = []

        for epoch in range(epochs):
            running_loss = 0.0
            valid_loss = 0.0

            total = 0
            correct = 0

            Net = GenderClassifier()
            Net.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(GenderClassifier.parameters(),lr=0.001,betas=(0.5,0.999))

            for i, data in enumerate(train,0):
                img, label = data

                img = img.to(self.device)
                label = label.to(self.device,dtype=torch.int64)
                
                optimizer.zero_grad()

                output = Net(img)
                _, predicted = torch.max(output,1)

                total += img.size(0)
                correct += (predicted == label.squeeze()).sum().item()

                loss = criterion(output.squeeze(),label.squeeze())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'[{epoch+1}/{epochs}] running_loss = {running_loss/len(train) :.3f}\t ### Accuracy = {100 * correct/total :.2f}% ### ')

            with torch.no_grad():

                total = 0
                correct = 0

                for i, data in enumerate(valid,0):
                    img, label = data

                    img = img.to(self.device)
                    label = label.to(self.device, dtype=torch.int64)

                    output = Net(img)

                    _, predicted = torch.max(output,1)

                    total += img.size(0)
                    correct += (predicted == label.squeeze()).sum().item()

                    val_loss = criterion(output.squeeze(),label.squeeze())

                    valid_loss += val_loss.item()

                running_loss_list.append(running_loss/len(train))    
                valid_loss_list.append(valid_loss/len(valid))

                bool_continue, early_stopping_val_loss_list = early_stopping_and_save_model(torch_model_name,valid_loss_list[-1],early_stopping_val_loss_list,early_stopping_patience)

                if not bool_continue:
                        print('{0}\nstop epoch : {1}\n{0}'.format('-' * 100, epoch - early_stopping_patience + 1))
                        break

                print(f'[{epoch+1}/{epochs}] val_loss = {valid_loss/len(valid) :.3f} \t ### Accuracy = {100 * correct/total :.2f}% ### \n')

        print('Finished Training')








