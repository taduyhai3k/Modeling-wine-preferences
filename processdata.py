import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import KFold


def GetDataTrainTest(list_of_numpydata, n_splits):
    list_data= {key: None for key,_ in list_of_numpydata.items()}
    FiveFold= KFold(n_splits= n_splits, shuffle = True)
    i = n_splits    
    while True:
        if i < n_splits:
            train = []
            test = []
            for key,values in list_data.items():
                index = next(values)
                train += [list_of_numpydata[key][i] for i in index[0]]
                test += [list_of_numpydata[key][i] for i in index[1]]
            yield train, test
            i = i+1
        else:
            for key, _ in list_data.items():
                list_data[key] = FiveFold.split(list_of_numpydata[key])
            i = 0     
class MyData(Dataset):
    def __init__(self, list_of_numpy_train) -> None:
        super().__init__()
        self.data = torch.tensor(np.array(list_of_numpy_train), dtype = torch.float32, requires_grad= False)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.data[index][-1].clone().detach().to(torch.long)
        feature = self.data[index][:-1].clone().detach().to(torch.float32).reshape(len(self.data[0])-1,1)
        return feature, label                   
                 