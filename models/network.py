import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LikeCategoryPredictor(nn.Module):
    def __init__(self, in_dim, image_in_dim, out_classes, convolutional=False,
                 resnet_weights=torchvision.models.resnet18(pretrained=True).state_dict()):
        super().__init__()

        if convolutional:
            self.feature = nn.Sequential(
                nn.Conv1d(in_channels=in_dim, out_channels=512, kernel_size=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
        else:
            self.feature = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=in_dim, out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=512, out_features=256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=256, out_features=128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
            )

        self.image_embedder = ImageEmbedder(in_dim=image_in_dim, out_dim=128, weights=resnet_weights)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 + 128, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            # nn.Linear(64, out_features=out_classes),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        data, image = x[0], x[1]  # Data, Image
        self.image_embedder.eval()

        # Data feature extraction
        data_features = self.feature(data)
        image_embeddings = self.image_embedder(image)
        # Concatenation
        concatenated = torch.cat([data_features, image_embeddings], dim=1)
        # Prediction
        out = self.linear(concatenated)

        return out


class ImageEmbedder(nn.Module):
    def __init__(self, in_dim, out_dim, weights=None):
        super().__init__()
        if weights is not None:
            self.model = torchvision.models.resnet18(pretrained=False)
            self.model.load_state_dict(weights)
        else:
            self.model = torchvision.models.resnet18()

        # Freeze weights
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Replace the last fc layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, input):
        output = self.model(input)
        return output