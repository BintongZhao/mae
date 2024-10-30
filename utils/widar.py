import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class Widar(Dataset):
    def __init__(self, root_path, num_classes, transform=None):
        self.root_path = root_path
        self.num_classes = num_classes
        self.transform = transform

        """
        self.scene_names = scene_names or ['Room1', 'Room2', 'Room3']
        self.scene_date_mapping = {
            'Room1': ['20181109', '20181112', '20181115', '20181116', '20181121', '20181130'],
            'Room2': ['20181117', '20181118', '20181127', '20181128', '20181204', '20181205', '20181208', '20181209'],
            'Room3': ['20181211']
        }
        """

        self.spectra = self.get_all_spectra()

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum_filepath = self.spectra[idx]
        spectrum, label = self.read_mat(spectrum_filepath), self.get_label_by_filepath(spectrum_filepath)
        spectrum, label = self.transform(spectrum), self.to_one_hot(label)
        return spectrum, label

    def get_all_spectra(self):
        spectra = []
        for root, _, files in os.walk(os.path.join(self.root_path, 'AOA-TOF')):
            for file in files:
                if file.endswith('.mat'):
                    spectra.append(os.path.join(root, file))
        return spectra

    def read_mat(self, filename):
        spectrum = scipy.io.loadmat(filename)['Pmusic'].astype(np.float32)
        return np.expand_dims(spectrum, axis=0)

    def get_label_by_filepath(self, filepath):
        filename = os.path.basename(filepath)
        return filename.split('_')[1]

    def to_one_hot(self, label):
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[int(label)] = 1.0
        return one_hot
