import pandas as pd
import os
from torch.utils.data import Dataset
from random import choice
import numpy as np

from datasets.utils import read_image


class TripletsCSVDataset(Dataset):
    '''
    Csv dataset representation (csv will be in RAM) for triplets
    '''

    def __init__(self,
                 csv_path: str,
                 image_prefix='',
                 path_transform=None,
                 transform=None,
                 return_triplets=True):
        '''
        :param csv_path: path to csv file with paths of images (one column)
        :param image_prefix: path prefix which will be added to paths of images in csv file
        :param path_transform: None or function for transform of path. Will be os.path.join(image_prefix,
         path_transform(image_path))
        :param transform: as like albumentations transform class or None
        :param return_triplets: if True, then return triplets images, else return image, label
        '''
        self.csv_path = csv_path
        self.image_prefix = image_prefix
        self.transform = transform
        self.path_transform = path_transform
        self.dt = pd.read_csv(csv_path)
        self.return_triplets = return_triplets
        images_per_classes = self.dt.iloc[:, 1].apply(lambda x: len(x.split(' '))).values
        self.idxs = np.zeros((images_per_classes.sum(), 2), dtype=np.int64)
        it = 0
        for i in range(len(images_per_classes)):
            self.idxs[it: it + images_per_classes[i], 0] = i
            self.idxs[it: it + images_per_classes[i], 1] = np.arange(images_per_classes[i])
            it += images_per_classes[i]

    def __len__(self):
        return len(self.idxs)

    def __read_image(self, image_id):
        image_path = image_id
        if self.path_transform is not None:
            image_path = self.path_transform(image_path)
        if self.image_prefix != '':
            image_path = os.path.join(self.image_prefix, image_path)
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __get_negative_id(self, anchor_id):
        negative_ids = list(range(anchor_id)) + list(range(anchor_id + 1, len(self.dt)))
        if len(negative_ids) == 0:
            raise ValueError(f'Dataset {self.csv_path} has only one label id')
        negative_id = choice(negative_ids)
        return choice(self.dt.iloc[negative_id].values[1].split(' '))

    def __get_positive_id(self, positive_image_ids, anchor_image_idx):
        positive_ids = list(range(anchor_image_idx)) + list(range(anchor_image_idx + 1, len(positive_image_ids)))
        if len(positive_ids) == 0:
            return positive_image_ids[anchor_image_idx]
        positive_image_idx = choice(positive_ids)
        return positive_image_ids[positive_image_idx]

    def __getitem__(self, idx):
        label_idx, image_idx = self.idxs[idx]
        row = self.dt.iloc[label_idx].values
        positive_image_ids = row[1].split(' ')
        anchor_id = positive_image_ids[image_idx]
        anchor_image = self.__read_image(anchor_id)
        if not self.return_triplets:
            return (anchor_image,), label_idx
        positive_id = self.__get_positive_id(positive_image_ids, image_idx)
        positive_image = self.__read_image(positive_id)

        negative_id = self.__get_negative_id(label_idx)
        negative_image = self.__read_image(negative_id)
        return (anchor_image, positive_image, negative_image), row[0]
