import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_dim=1024, dropout=0.5, device=None):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_shape[0] * input_shape[1]
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.classifier.to(self.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    input_shape = (128, 768)
    num_classes = 10
    classifier = MLPClassifier(input_shape, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = torch.randn(64, 128, 768).to(device)

    outputs = classifier(features)

    print("outputs shape:", outputs.shape)


if __name__ == '__main__':
    main()
