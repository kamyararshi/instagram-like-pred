import torch
import torch.nn as nn
import torch.nn.functional as F

class LikeCategoryPredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
         

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 10),
            #nn.Softmax(10)
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.feature(x)
        out = self.linear(x)
        #out = self.softmax(out)

        return out
    
# TODO: Later we do regression on the like amount
class Predictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        None

    def forwrd(self, x):
        out = x
        return out