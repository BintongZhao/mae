import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class XRF55(Dataset):
    def __init__(self, root_path, num_classes=55, scene_names=['Scene_1'], transform=None):
        self.root_path = root_path
        self.num_classes = num_classes
        self.scene_names = scene_names or ['Scene_1', 'Scene_2', 'Scene_3', 'Scene_4']
        self.transform = transform
        self.receiver_names = ['lb', 'rb', 'lf']
        self.scene_person_mapping = {
            'Scene_1': [f"{i:02}" for i in range(1, 31)],
            'Scene_2': ['05', '24', '31'],
            'Scene_3': ['06', '07', '23'],
            'Scene_4': ['03', '04', '13']
        }

        self.partial_filenames = self.get_partial_filenames()
        self.filepaths = self.get_filepaths()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        spectra, label = self.get_item(idx)
        spectra, label = self.transform(spectra), self.to_one_hot(label)
        return spectra, label

    def to_one_hot(self, label):
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[int(label)] = 1.0
        return one_hot

    def get_partial_filenames(self):
        parent_path = os.path.join(self.root_path, 'AOA-TOF', 'Scene_1', 'lb', '01')
        partial_filenames = []
        for filename in os.listdir(parent_path):
            if os.path.isfile(os.path.join(parent_path, filename)):
                filename_without_extension, _ = os.path.splitext(filename)
                parts = filename_without_extension.split('_')
                partial_filenames.append('_' + '_'.join(parts[1:]))
        return partial_filenames

    def get_filepaths(self):
        filepaths = []
        for scene_name in self.scene_names:
            for person_name in self.scene_person_mapping[scene_name]:
                parent_path = os.path.join(self.root_path, 'AOA-TOF', scene_name, 'lb', person_name)
                filepaths.extend([
                    os.path.join(parent_path, f"{person_name}{partial_filename}")
                    for partial_filename in self.partial_filenames
                ])
        return filepaths

    def replace_receiver_name(self, filepath, new_receiver_name):
        return os.path.normpath(filepath.replace('lb', new_receiver_name))

    def get_item_all_filepaths(self, index):
        prefix = self.filepaths[index]
        filepaths = [
            self.replace_receiver_name(f"{prefix}_{i}.mat", receiver_name)
            for i in range(1, 101)
            for receiver_name in self.receiver_names
        ]
        return filepaths

    def get_label_by_filepath(self, filepath):
        filename = os.path.basename(filepath)
        return filename.split('_')[1]

    def read_mat(self, filename):
        spectrum = scipy.io.loadmat(filename)['Pmusic'].astype(np.float32)
        return np.expand_dims(spectrum, axis=0)

    def get_item(self, index):
        all_filepaths = self.get_item_all_filepaths(index)
        spectra = np.vstack([self.read_mat(filepath) for filepath in all_filepaths])
        label = self.get_label_by_filepath(all_filepaths[0])
        return spectra, label
