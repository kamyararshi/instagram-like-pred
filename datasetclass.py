import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

class InstagramUserData(Dataset):
    def __init__(self, data_path, device='cpu', train=True) -> None:
        super().__init__()
        self.dataframe_orig = pd.read_csv(data_path).drop('Unnamed: 0', axis=1)
        self.device = device
        self.labels = self.dataframe_orig['numberLikesCategory']
        self.data = self.dataframe_orig.drop(['numberLikesCategory'], axis=1)
        data_train, data_test, labels_train, labels_test = train_test_split(
            self.data, self.labels, test_size=.1, random_state=42
        )

        if train:
            self.dataframe = data_train
            self.labels = labels_train
        else:
            self.dataframe = data_test
            self.labels = labels_test

    def __getitem__(self, index) -> torch.Tensor:
        return torch.tensor(self.dataframe.iloc[index], dtype=torch.float32, device=self.device).reshape(1, -1)\
        , torch.tensor(self.labels.iloc[index], dtype=torch.float32, device=self.device).reshape(1, -1)
    def __len__(self):
        return len(self.dataframe)
