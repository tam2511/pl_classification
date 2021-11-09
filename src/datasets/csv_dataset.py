import pandas as pd
import os
from torch.utils.data import Dataset

from datasets.utils import read_image


class CSVDataset(Dataset):
    '''
    Csv dataset representation (csv will be in RAM)
    '''

    def __init__(self, csv_path: str, image_prefix='', transform=None, return_label=True):
        '''
        :param csv_path: path to csv file with paths of images ang labels in one hot encoding
        :param image_prefix: path prefix which will be added to paths of images in csv file
        :param transform: as like albumentations transform class or None
        :param return_label: if True dataset will return labels
        '''
        self.image_prefix = image_prefix
        self.transform = transform
        self.return_label = return_label
        self.dt = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.dt)

    def read_image__(self, idx, debug=False):
        row = self.dt.iloc[idx].values
        image_path = row[0] if self.image_prefix == '' else os.path.join(self.image_prefix, row[0])
        image = read_image(image_path)
        if self.transform and not debug:
            image = self.transform(image=image)['image']
        if not self.return_label:
            return image
        label = row[1:].astype('int32')
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
