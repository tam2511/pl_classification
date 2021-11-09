import os
import numpy as np
from torch.utils.data import Dataset
from datasets.utils import check_image_load as check_image_load_f, read_image, get_label


class DirDataset(Dataset):
    '''
    Dataset implementation for images in directory on disk (stored images paths and labels in RAM).
    Require root_path/label_name/.../image_path structure.
    '''

    def __init__(self, root_path: str, transform=None, return_label=True, check_image_load=False):
        '''
        :param root_path: path of directory with images
        :param transform: as like albumentations transform class or None
        :param return_label: if True dataset will return labels
        :param check_image_load: if True in __init__ will check is image correct
        '''
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
                label_name = get_label(os.path.relpath(file_name, self.root_path))
                self.labels.append(self.label_names == label_name)
        if self.return_label:
            self.labels = np.stack(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def read_image__(self, idx, debug=False):
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        if self.transform and not debug:
            image = self.transform(image=image)['image']
        if not self.return_label:
            return image
        label = self.labels[idx]
        return image, label

    def __getitem__(self, idx):
        return self.read_image__(idx)

    def debug(self, idx):
        '''
        Get image (and label) without transform
        :param idx: int
        :return: image (or image, label)
        '''
        return self.read_image__(idx, debug=True)
