import numpy as np
from torch.utils.data import Dataset
import torchvision.datasets as dset

class CustomDataset(Dataset):
    def __init__(self,data_path,csv_path,transform=None):
        self.data_path = data_path
        self.csv_path = csv_path
        self.label_arr = np.asarray(pd.read_csv(self.csv_path))
    
    def __len__(self):
        return len(self.label_arr)
    
    def __getitem__(self,idx):
        image = dset.ImageFolder(root=self.data_path,transform=transform)[idx][0]
        label = self.label_arr[idx]

        return (image, label)

def early_stopping_and_save_model(torch_model_name,input_vali_loss, early_stopping_val_loss_list, early_stopping_patience):

    if len(early_stopping_val_loss_list) != early_stopping_patience:
        early_stopping_val_loss_list = [99.99 for _ in range(early_stopping_patience)]

    early_stopping_val_loss_list.append(input_vali_loss)
    if input_vali_loss < min(early_stopping_val_loss_list[:-1]):
        torch.save(GenderClassifier, '{}/{}'.format('drive/My Drive', torch_model_name))
        early_stopping_val_loss_list.pop(0)

        return True, early_stopping_val_loss_list

    elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list):
        return False, early_stopping_val_loss_list