import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim):
        self.conv1 = nn.Conv2d()

    def forward(self, x):
        out=x
        return out
    

class Predictor(nn.Module):
    def __init__(self, in_dim):
        self.ftrs_extrct = FeatureExtractor(in_dim)
        self.fc1 = nn.fc()
        self.softmax = nn.softmax()

    def forwrd(self, x):
        x = self.ftrs_extrct(x)
        x = self.fc(x)
        out = self.softmax(x)
        return out