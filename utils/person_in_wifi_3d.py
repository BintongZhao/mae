import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class PersonInWifi3d(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform

        self.filepaths = self.get_filepaths()

    def __len__(self):
        return len(self.filepaths) // 3

    def __getitem__(self, idx):
        spectra, pose = self.get_item(idx * 3)

        if self.transform:
            spectra = self.transform(spectra)
        pose = torch.tensor(pose, dtype=torch.float32)

        return spectra, pose

    def get_filepaths(self):
        filepaths = []
        for dirpath, _, filenames in os.walk(os.path.join(self.root_path, 'AOA-TOF')):
            for filename in filenames:
                if filename.endswith('.mat'):
                    filepaths.append(os.path.join(dirpath, filename))
        return filepaths

    def read_mat(self, filename):
        spectrum = scipy.io.loadmat(filename)['Pmusic'].astype(np.float32)
        return np.expand_dims(spectrum, axis=0)

    def get_pose_by_filepath(self, filepath):
        parent_path, filename = os.path.split(filepath.replace('AOA-TOF', 'POSE'))
        pose_filepath = os.path.join(parent_path, f'{filename[:-7]}.npy')
        return np.load(pose_filepath).astype(np.float32)

    def get_item(self, index):
        spectra = np.vstack([self.read_mat(filepath) for filepath in self.filepaths[index:index + 3]])
        pose = self.get_pose_by_filepath(self.filepaths[index])
        return spectra, pose
