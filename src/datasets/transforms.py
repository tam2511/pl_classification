from albumentations.augmentations import *
from albumentations.core.composition import *
from albumentations.pytorch.transforms import *

available_transform_names = ['Blur', 'CLAHE', 'ChannelDropout', 'ChannelShuffle', 'ColorJitter', 'Downscale', 'Emboss',
                             'Equalize', 'FDA', 'FancyPCA', 'FromFloat', 'GaussNoise', 'GaussianBlur', 'GlassBlur',
                             'HistogramMatching', 'HueSaturationValue', 'ISONoise', 'ImageCompression', 'InvertImg',
                             'MedianBlur',
                             'MotionBlur', 'MultiplicativeNoise', 'Normalize', 'PixelDistributionAdaptation',
                             'Posterize',
                             'RGBShift', 'RandomBrightnessContrast', 'RandomFog', 'RandomGamma', 'RandomRain',
                             'RandomShadow',
                             'RandomSnow', 'RandomSunFlare', 'RandomToneCurve', 'Sharpen', 'Solarize', 'Superpixels',
                             'TemplateTransform', 'ToFloat', 'ToGray', 'ToSepia', 'Affine', 'CenterCrop',
                             'CoarseDropout', 'Crop',
                             'CropAndPad', 'CropNonEmptyMaskIfExists', 'ElasticTransform', 'Flip', 'GridDistortion',
                             'GridDropout', 'HorizontalFlip', 'Lambda', 'LongestMaxSize', 'MaskDropout', 'NoOp',
                             'OpticalDistortion', 'PadIfNeeded', 'Perspective', 'PiecewiseAffine', 'RandomCrop',
                             'RandomCropNearBBox', 'RandomGridShuffle', 'RandomResizedCrop', 'RandomRotate90',
                             'RandomScale',
                             'RandomSizedBBoxSafeCrop', 'RandomSizedCrop', 'Resize', 'Rotate', 'SafeRotate',
                             'ShiftScaleRotate',
                             'SmallestMaxSize', 'Transpose', 'VerticalFlip', 'ToTensorV2']

available_composition_names = ['Compose', 'OneOf', 'PerChannel', 'Sequential', 'SomeOf']


def create_transform(transforms: list) -> BaseCompose:
    '''
    Create Compose of transform from albumentation lib
    :param transforms: list of dicts
    :return: list of transforms
    '''

    def create_transform_(transforms: list) -> list:
        result = []
        for transform in transforms:
            if transform['name'] in available_composition_names:
                result.append(
                    eval(transform['name'])(transforms=create_transform_(transform['transforms']),
                                            **transform['kwargs']))
            elif transform['name'] in available_transform_names:
                result.append(eval(transform['name'])(**transform['kwargs']))
        return result

    return Compose(create_transform_(transforms))
