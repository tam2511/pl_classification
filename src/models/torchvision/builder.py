from torchvision.models.mobilenet import *
from torchvision.models.shufflenetv2 import *
from torchvision.models.squeezenet import *
from torchvision.models.mnasnet import *
from torchvision.models.alexnet import *
from torchvision.models.densenet import *
from torchvision.models.googlenet import *
from torchvision.models.inception import *
from torchvision.models.resnet import *
from torchvision.models.vgg import *

available_names = [
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'shufflenet_v2_x1_0', 'squeezenet1_0',
    'mnasnet1_0', 'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet', 'inception_v3',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'wide_resnet50_2',
    'wide_resnet101_2'
]


def create_model(model_name, kwargs):
    if model_name in available_names:
        return eval(model_name)(**kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(model_name))
