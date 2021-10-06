import torch.nn as nn
from models.torchvision.builder import create_model as torchvision_create_model


def create_model(model_name: str, kwargs: dict) -> nn.Module:
    '''
    create torch.nn.Module object model
    :param model_name: str
    :param kwargs: dict
    :return: torch.nn.Module object
    '''
    domen = model_name.split('.')[0]
    name = '.'.join(model_name.split('.')[1:])
    if domen == 'torchvision':
        return torchvision_create_model(name, kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(domen))
