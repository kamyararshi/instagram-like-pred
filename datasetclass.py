import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

from utils import *

import numpy as np
import pandas as pd
import os

class InstagramUserData(Dataset):
    def __init__(self, data_path, seperator=';', device='cpu', train=True) -> None:
        super().__init__()
        self.dataframe_orig = pd.read_csv(data_path, sep=seperator).drop('Unnamed: 0', axis=1)
        self.device = device
        self.image_paths = self.dataframe_orig['path']
        self.dataframe_orig['numberLikes'] = self.dataframe_orig['numberLikes'].fillna(self.dataframe_orig['numberLikes'].mean())
        self.labels = self.dataframe_orig['numberLikes']

        # Change Weekday to numerical later and use it as a feature
        self.data = self.dataframe_orig.drop(['numberLikesCategory', 'descriptionProcessed', 'path', 'numberLikes', 'url', 'alias', 'weekday'], axis=1) # We may use the descriptin to extract embeddigs latter
        # Split train-test
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
        # example = {}
        data = self.dataframe.iloc[index]
        labels = self.labels.iloc[index]
        path = self.image_paths[index]

        # Create dict
        data = torch.tensor(data, dtype=torch.float32, device=self.device).reshape(-1, 1).to(self.device).to(torch.float32)
        try:
            image = read_image(path=path).to(self.device).to(torch.float32) #TODO: Need a collate function to have all images with the same size
        except Exception as e:
            print(e, "in", path)
            image = torch.tensor(999)
        label = torch.tensor(labels, dtype=torch.float32, device=self.device)
        return data, image, label
    
    def __len__(self):
        return len(self.dataframe)
    
    def _return_loader(self, batch_size, shuffle, num_workers, collate_fn):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
