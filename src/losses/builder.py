import torch.nn as nn
from losses.multiclass.focalloss import FocalLoss
from losses.multilabel.assymetric_loss import AsymmetricLoss, AsymmetricLossOptimized


def create_loss(loss_name: str, multilabel: bool, kwargs: dict) -> nn.Module:
    '''
    Create torch.nn.Module object loss function
    :param loss_name: name of loss function
    :param multilabel: multilabel loss flag
    :param kwargs: dict of loss function arguments
    :return: torch.nn.Module
    '''
    if multilabel:
        return create_multilabel_loss_(loss_name=loss_name, kwargs=kwargs)
    else:
        return create_singlelabel_loss_(loss_name=loss_name, kwargs=kwargs)


def create_singlelabel_loss_(loss_name: str, kwargs: dict) -> nn.Module:
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'FocalLoss':
        return FocalLoss(**kwargs)
    else:
        raise NotImplementedError('{} not implemented for singlelabel case'.format(loss_name))


def create_multilabel_loss_(loss_name: str, kwargs: dict) -> nn.Module:
    if loss_name == 'BCELoss':
        return nn.BCELoss(**kwargs)
    elif loss_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'AsymmetricLoss':
        return AsymmetricLoss(**kwargs)
    elif loss_name == 'AsymmetricLossOptimized':
        return AsymmetricLossOptimized(**kwargs)
    else:
        raise NotImplementedError('{} not implemented for multilabel case'.format(loss_name))
