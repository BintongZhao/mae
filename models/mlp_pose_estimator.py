import torch
import torch.nn as nn


class MLPPoseEstimator(nn.Module):
    def __init__(self, input_shape, hidden_dim=1024, dropout=0.5, device=None, joint_num=17, dimension=3):
        super(MLPPoseEstimator, self).__init__()
        self.input_dim = input_shape[0] * input_shape[1]
        self.hidden_dim = hidden_dim
        self.joint_num = joint_num
        self.dimension = dimension
        self.output_dim = self.joint_num * self.dimension
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pose_estimator = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.pose_estimator.to(self.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.pose_estimator(x)
        return y.view(y.shape[0], self.joint_num, self.dimension)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_estimator = MLPPoseEstimator((128, 768), device=device)

    features = torch.randn(64, 128, 768).to(device)

    # torch.Size([64, 17, 3])
    outputs = pose_estimator(features)

    print("outputs shape:", outputs.shape)


if __name__ == '__main__':
    main()
