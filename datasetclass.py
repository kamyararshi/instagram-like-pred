import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import re


def parse_array(s):
    # Remove the brackets and split the string into elements
    s = s[1:-1].split()
    # Convert each string to a float and create a numpy array
    return np.array([float(x) for x in s])


class InstagramUserData(Dataset):
    def __init__(self, data_path, device='cpu', train=True) -> None:
        super().__init__()
        self.dataframe_orig = pd.read_csv(data_path).drop('Unnamed: 0', axis=1)
        self.device = device
        self.labels = self.dataframe_orig['numberLikesCategory'] - 1  # Make labels to be from 0-9
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
        # Get the vector and real number features
        vector_feature_str = self.dataframe.iloc[index]['descriptionVector']
        vector_feature = parse_array(vector_feature_str)
        other_features = self.dataframe.drop(columns='descriptionVector').iloc[index]

        # Convert the vector feature to a tensor
        vector_tensor = torch.tensor(vector_feature, dtype=torch.float32, device=self.device)

        # Convert the other features to a tensor
        other_tensor = torch.tensor(other_features.values, dtype=torch.float32, device=self.device)

        # Concatenate the vector and other features into a single tensor
        data_tensor = torch.cat([vector_tensor, other_tensor])
        data_tensor = data_tensor.unsqueeze(-1)

        # Get the label
        label = torch.tensor(self.labels.iloc[index], dtype=torch.float32, device=self.device).to(torch.long)
        return data_tensor, label

    def __len__(self):
        return len(self.dataframe)
