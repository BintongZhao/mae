import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MLPClassifier, self).__init__()
        input_dim = input_shape[0] * input_shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    input_shape = (128, 768)
    num_classes = 10
    classifier = MLPClassifier(input_shape, num_classes)

    features = torch.randn(64, 128, 768)

    outputs = classifier(features)

    print("outputs shape:", outputs.shape)


if __name__ == '__main__':
    main()
