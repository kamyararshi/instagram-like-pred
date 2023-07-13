import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LikeCategoryPredictor(nn.Module):
    def __init__(self, in_dim, image_in_dim, out_classes, convolutional=False, resnet_weights=torchvision.models.ResNet18_Weights):
        super().__init__()

        if convolutional:
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
        else:
            self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            )  

        
        # Output shape is 512
        self.image_embedder = ImageEmbedder(in_dim=image_in_dim, weights=resnet_weights)
        #TODO: Change the linear layer depending on the shape of the concatenated tensor
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512+32, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, out_features=out_classes),
            #nn.Softmax(10)
        )


    def forward(self, x):
        data, image = x[0], x[1] # Data, Image
        self.image_embedder.eval()
        
        # Data feature extraction
        data_features = self.feature(data)
        image_embeddings = self.image_embedder(image)
        # Concatentaion
        concatenated = torch.concatenate([data_features, image_embeddings], dim=1)
        # prediciton
        out = self.linear(concatenated)

        return out
    
# TODO: Later we do regression on the like amount
class Predictor(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        None

    def forward(self, x):
        out = x
        return out
    
class ImageEmbedder(nn.Module):
    def __init__(self, in_dim, weights=None) -> None:
        super().__init__()
        if weights is not None:
            self.model = torchvision.models.resnet18(weights)

        else:
            self.model = torchvision.models.resnet18()

        # Remove the last fc layer
        self.model.fc = nn.Identity()

    def forward(self, input):
        output = self.model(input)
        return output
    
