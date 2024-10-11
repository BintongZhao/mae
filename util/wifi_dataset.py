import os
import numpy as np
import scipy
from torch.utils.data import Dataset


class WiFiDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.file_paths = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.mat')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = scipy.io.loadmat(file_path)
        sample = data['Pmusic'].astype(np.float32).reshape(1, 181, 21)
        sample = self.transform(sample).permute(1, 2, 0)

        return sample
