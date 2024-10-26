import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EnvDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform or transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float())
        ])
        self.data_file_paths, self.labels = self._load_data()

    def _load_data(self):
        data_file_paths, labels = [], []

        for env in os.listdir(self.root_dir):
            env_dir = os.path.join(self.root_dir, env)

            ground_truth = np.load(os.path.join(env_dir, 'ground_truth.npy'))
            data_files = [f for f in os.listdir(env_dir) if f != 'ground_truth.npy']
            for file_name in data_files:
                file_path = os.path.join(env_dir, file_name)
                data_file_paths.append(file_path)
                labels.append(ground_truth)

        return data_file_paths, labels

    def __len__(self):
        return len(self.data_file_paths)

    def __getitem__(self, idx):
        data, label = np.load(self.data_file_paths[idx]), self.labels[idx]
        data, label = np.expand_dims(data, axis=0), np.expand_dims(label, axis=0)
        return self.transform(data), self.transform(label)


def get_dataloader(root_dir, batch_size, mode='train', transform=None, shuffle=True, num_workers=4):
    dataset = EnvDataset(root_dir=root_dir, mode=mode, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    root_dir = '/home/xinglibao/WorkSpace/datasets/env'
    batch_size = 8

    train_loader = get_dataloader(root_dir, batch_size, mode='train')
    val_loader = get_dataloader(root_dir, batch_size, mode='val', shuffle=False)
    test_loader = get_dataloader(root_dir, batch_size, mode='test', shuffle=False)

    for data, label in train_loader:
        print(data.shape, label.shape)

    for data, label in val_loader:
        print(data.shape, label.shape)

    for data, label in test_loader:
        print(data.shape, label.shape)
