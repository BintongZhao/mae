import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class Widar(Dataset):
    def __init__(self, root_path, num_classes, scene_names=['Room1'], transform=None):
        self.root_path = root_path
        self.num_classes = num_classes
        self.transform = transform
        self.scene_names = scene_names or ['Room1', 'Room2', 'Room3']
        self.scene_date_mapping = {
            'Room1': ['20181109', '20181112', '20181115', '20181116', '20181121', '20181130'],
            'Room2': ['20181117', '20181118', '20181127', '20181128', '20181204', '20181205', '20181208', '20181209'],
            'Room3': ['20181211']
        }

        self.filepaths = self.get_all_filepaths()

    def __len__(self):
        return len(self.filepaths) // 240

    def __getitem__(self, idx):
        spectra, label = self.get_item(idx * 240)

        if self.transform:
            spectra = self.transform(spectra)
        label = self.to_one_hot(label)

        return spectra, label

    def to_one_hot(self, label):
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[int(label)] = 1.0
        return one_hot

    def get_filepaths(self, root_path):
        filepaths = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith('.mat'):
                    filepaths.append(os.path.join(dirpath, filename))
        return filepaths

    def get_all_filepaths(self):
        filepaths = []
        for scene_name in self.scene_names:
            for date in self.scene_date_mapping[scene_name]:
                root_path = os.path.join(self.root_path, 'AOA-TOF', date)
                filepaths.extend(self.get_filepaths(root_path))
        return filepaths

    def read_mat(self, filename):
        spectrum = scipy.io.loadmat(filename)['Pmusic'].astype(np.float32)
        return np.expand_dims(spectrum, axis=0)

    def get_label_by_filepath(self, filepath):
        filename = os.path.basename(filepath)
        return filename.split('_')[1]

    def get_item(self, index):
        spectra = np.vstack([self.read_mat(filepath) for filepath in self.filepaths[index:index + 240]])
        label = self.get_label_by_filepath(self.filepaths[index])
        return spectra, label
