import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dims=(512, 256),
                 img_size=(128, 16), patch_size=4, device=None):
        super(ResNetClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.height = img_size[0] // patch_size
        self.width = img_size[1] // patch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.hidden_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dims[1], 3, kernel_size=1),
            nn.ReLU()
        )

        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

        self.to(self.device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], self.feature_dim, self.height, self.width)
        x = self.conv_layers(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.resnet(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dim = 768
    num_classes = 10
    classifier = ResNetClassifier(feature_dim, num_classes)

    features = torch.randn(64, 32 * 4, feature_dim).to(device)

    outputs = classifier(features)

    print("outputs shape:", outputs.shape)


if __name__ == '__main__':
    main()
