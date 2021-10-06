import os
import numpy as np
import logging
from torch.utils.data import Dataset
from datasets.utils import check_image_load as check_image_load_f
import cv2


class DirDataset(Dataset):
    def __init__(self, root_path, transform, return_label=True, check_image_load=False):
        self.root_path = root_path
        self.transform = transform
        self.return_label = return_label
        self.check_image_load = check_image_load
        self.image_paths = []
        self.label_names = []
        self.labels = []
        self.load__()

    def load__(self):
        self.label_names = np.array([os.path.basename(os.path.join(self.root_path, dir_name)) for dir_name in
                                     os.listdir(self.root_path) if
                                     os.path.isdir(os.path.join(self.root_path, dir_name))])
        for root, _, files in os.walk(self.root_path):
            for file_name in files:
                file_name = os.path.join(root, file_name)
                if self.check_image_load and not check_image_load_f(file_name):
                    continue
                self.image_paths.append(file_name)
                if not self.return_label:
                    continue
                label_name = os.path.dirname(os.path.relpath(file_name, self.root_path))
                self.labels.append(self.label_names == label_name)
        if self.return_label:
            self.labels = np.stack(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.transform:
            image = self.transform(image=image)['image']
        if not self.return_label:
            return image
        label = self.labels[idx]
        return image, label

    def debug(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if not self.return_label:
            return image
        label = self.labels[idx]
        return image, label
