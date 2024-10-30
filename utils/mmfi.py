import os
import re
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class MMFI(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.spectrum_path = os.path.join(self.root_path, 'AOA_TOF')
        self.pose_path = os.path.join(self.root_path, 'POSE')
        self.transform = transform

        self.filepaths = self.get_filepaths()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        spectrum_filepath = self.filepaths[idx]
        spectrum, pose = self.read_mat(spectrum_filepath), self.get_pose_by_filepath(spectrum_filepath)

        if self.transform:
            spectrum = self.transform(spectrum)
        pose = torch.tensor(pose, dtype=torch.float32)

        return spectrum, pose

    def get_filepaths(self):
        filepaths = []
        for dirpath, _, filenames in os.walk(self.spectrum_path):
            for filename in filenames:
                if filename.endswith('.mat'):
                    filepaths.append(os.path.join(dirpath, filename))
        return filepaths

    def read_mat(self, filename):
        spectrum = scipy.io.loadmat(filename)['Pmusic'].astype(np.float32)
        return np.expand_dims(spectrum, axis=0)

    def get_pose_by_filepath(self, filepath):
        match = re.search(r"(E\d+).*(S\d+).*(A\d+)", filepath)
        filename = f'{match.group(1)}-{match.group(2)}-{match.group(3)}.npy'
        pose_filepath = os.path.join(self.pose_path, filename)
        return np.load(pose_filepath).astype(np.float32)
