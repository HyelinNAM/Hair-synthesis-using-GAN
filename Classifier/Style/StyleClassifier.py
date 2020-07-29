import os
import random
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset,random_split,Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Style_utils import CustomDataset,early_stopping_and_save_model
from Style_models import StyleClassifier

class StyleClassifier(self):
    def __init__(self, mode = 'train'):
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        manualSeed = 999
        print("Random seed:",manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        if self.mode == 'train':
            data_F_path = 'drive/My Drive/Data/Face' # Face
            label_F_path = 'drive/My Drive/Data/face.csv'
            data_C_path = 'drive/My Drive/Data/CelebA-HQ'# CelebA-HQ
            label_C_path = 'drive/My Drive/Data/CelebA.csv'

            self.path = {'data_F_path':data_F_path,'label_F_path':label_F_path,'data_C_path':data_C_path, 'label_C_path':label_C_path}

        elif self.mode == 'test':
            self.path = 'drive/My Drive/Data/Face'

    def __call__(self):

        if self.mode == 'train':
            print("Making Custom dataset...")
            Face = self.mk_dataset()
            train_sampler, valid_sampler = self.data_split(valid_ratio=0.2)

            train = DataLoader(Face,sampler=train_sampler,batch_size=16,drop_last=True)
            valid = DataLoader(Face,sampler=valid_sampler,batch_size=16,drop_last=True)

            print(f'Number of Dataset: train-{len(train)} valid-{len(valid)}')

            self.train(train,valid)

        elif self.mode == 'test':
            print("Making Custom dataset...")
            Face = self.mk_dataset()

            test = DataLoader(Face,batch_size=16)

            self.test(test)

            return results

        else
            raise Exception('Wrong mode!')
        
    def mk_dataset(self):
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

        if self.mode == 'train':
            Face_F = CustomDataset(self.path['data_F_path'],self.path['label_F_path'],transform)
            Face_F_aug = CustomDataset(self.path['data_F_path'],self.path['label_F_path'],transform2)
            Face_C = CustomDataset(self.path['data_C_path'],self.path['label_C_path'],transform)
            Face_C_aug = CustomDataset(self.path['data_C_path'],self.path['label_C_path'],transform2)
            
            Face = ConcatDataset([Face_F,Face_F_aug,Face_C,Face_C_aug])

            return Face

        else:
            Face = CustomDataset(self.path,transform)

            return Face

    def data_split(valid_ratio):
        np.random.seed(777)

        size = len(self.Face)
        indices = list(range(size))

        np.random.shuffle(indices)

        split_valid = int(np.floor(size*valid_ratio))

        train_indices, val_indices = indices[split_valid:],indices[:split_valid]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        return train_sampler,valid_sampler

    def train(train,valid,epochs=200):

        # For ealry stopping
        torch_model_name = "StyleClassifier"
        early_stopping_patience = 20

        print("Start Training...!")

        running_loss_list = []
        valid_loss_list = []
        early_stopping_val_loss_list = []

        Net = StyleClassifier()
        Net.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(StyleClassifier.parameters(),lr=0.001,betas=(0.5,0.999))

        for epoch in range(epochs):

            running_loss = 0.0
            valid_loss = 0.0

            total = 0
            correct1 = 0
            correct2 = 0
            correct3 = 0

            for i, data in enumerate(train,0):
                img, label = data

                img = img.to(self.device)
                label1 = torch.tensor(list(map(lambda label: label[0:1], label)))
                torch.flatten(label1)
                label2 = torch.tensor(list(map(lambda label: label[1:2], label)))
                torch.flatten(label2)
                label3 = torch.tensor(list(map(lambda label: label[2:3], label)))
                torch.flatten(label3)

                label1 = label1.to(self.device, dtype=torch.int64)
                label2 = label2.to(self.device, dtype=torch.int64)
                label3 = label3.to(self.device, dtype=torch.int64)
                
                optimizer.zero_grad()

                output1, output2, output3 = Net(img)
                _, predicted1 = torch.max(output1,1)
                _, predicted2 = torch.max(output2,1)
                _, predicted3 = torch.max(output3,1)

                total += img.size(0)
                correct1 += (predicted1 == label1.squeeze()).sum().item()
                correct2 += (predicted2 == label2.squeeze()).sum().item()
                correct3 += (predicted3 == label3.squeeze()).sum().item()

                loss1 = criterion(output1.squeeze(),label1.squeeze())
                loss2 = criterion(output2.squeeze(),label1.squeeze())
                loss3 = criterion(output3.squeeze(),label1.squeeze())

                loss = (loss1 + loss2 + loss3)/3

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'[{epoch+1}/{epochs}] running_loss = {running_loss/len(train) :.3f}\t ### Accuracy = {100 * correct/total :.2f}% ### ')

            with torch.no_grad():

                total = 0
                correct1 = 0
                correct2 = 0
                correct3 = 0

                for i, data in enumerate(valid,0):
                    img, label = data

                    img = img.to(self.device)
                    label1 = torch.tensor(list(map(lambda label: label[0:1], label)))
                    torch.flatten(label1)
                    label2 = torch.tensor(list(map(lambda label: label[1:2], label)))
                    torch.flatten(label2)
                    label3 = torch.tensor(list(map(lambda label: label[2:3], label)))
                    torch.flatten(label3)

                    output1, output2, output3 = Net(img)
                    _, predicted1 = torch.max(output1,1)
                    _, predicted2 = torch.max(output2,1)
                    _, predicted3 = torch.max(output3,1)

                    total += img.size(0)
                    correct1 += (predicted1 == label1.squeeze()).sum().item()
                    correct2 += (predicted2 == label2.squeeze()).sum().item()
                    correct3 += (predicted3 == label3.squeeze()).sum().item()

                    loss1 = criterion(output1.squeeze(),label1.squeeze())
                    loss2 = criterion(output2.squeeze(),label1.squeeze())
                    loss3 = criterion(output3.squeeze(),label1.squeeze())

                    loss = (loss1 + loss2 + loss3)/3

                    valid_loss += loss.item()

                running_loss_list.append(running_loss/len(train))    
                valid_loss_list.append(valid_loss/len(valid))

                bool_continue, early_stopping_val_loss_list = early_stopping_and_save_model(torch_model_name,valid_loss_list[-1],early_stopping_val_loss_list,early_stopping_patience)

                if not bool_continue:
                        print('{0}\nstop epoch : {1}\n{0}'.format('-' * 100, epoch - early_stopping_patience + 1))
                        break

                print(f'[{epoch+1}/{epochs}] val_loss = {valid_loss/len(valid) :.3f} \t ### Accuracy = {100 * correct/total :.2f}% ### \n')

        print('Finished')
    
    def test(test):
        
        Net = StyleClassifier()
        Net = torch.load('drive/My Drive/model/StyleClassifier')
        Net.to(self.device)

        Net.eval()

        color_results=[]
        bang_results=[]
        style_results=[]

        print("Start Testing...!")

        for i, data in enumerate(test,0):

            img,_ = data
            img = img.to(self.device)

            color, bang, style = Net(img)
            
            _, color = torch.max(color,1)
            _, bang = torch.max(bang,1)
            _, style = torch.max(style,1)

            color_results.append(color.tolist())
            bang_results.append(bang.tolist())
            style_results.append(style.tolist())


        with open('drive/My Drive/results/Color.json','w') as f:
            json.dump(color_results,f)

        with open('drive/My Drive/results/Bang.json','w') as f:
            json.dump(bang_results,f)

        with open('drive/My Drive/results/Style.json','w') as f:
            json.dump(style_results,f)

        print('Finished')

        return color_results,bang_results,style_results


if __name__ == "__main__":
    model = StyleClassifier()
    color_results,bang_results,style_results = model(mode = 'test') # or mode = 'train'










