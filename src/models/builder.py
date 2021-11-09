import torch.nn as nn
from models.torchvision.builder import create_model as torchvision_create_model


def create_model(model_name: str, model_domen: str, kwargs: dict) -> nn.Module:
    '''
    create torch.nn.Module object model
    :param model_name: str
    :param model_domen: str
    :param kwargs: dict
    :return: torch.nn.Module object
    '''
    if model_domen == 'torchvision':
        return torchvision_create_model(model_name, kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(model_name))
