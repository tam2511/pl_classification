from datasets.csv_dataset import CSVDataset
from datasets.dir_dataset import DirDataset


def create_dataset(dataset_type: str, transform, kwargs: dict):
    '''
    Create Dataset object
    :param dataset_type: type of dataset
    :param transform: transform object
    :param kwargs: dict of params
    :return:
    '''
    if dataset_type == 'csv':
        return CSVDataset(transform=transform, **kwargs)
    elif dataset_type == 'directory':
        return DirDataset(transform=transform, **kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_type))
