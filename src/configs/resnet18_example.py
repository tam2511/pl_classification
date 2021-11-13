from easydict import EasyDict

cfg = EasyDict()

cfg.trainer_kwargs = {
    'gpus': 1
}

cfg.model = {
    'domen': 'torchvision',
    'name': 'resnet18',
    'kwargs': {
        'pretrained': True,
        'num_classes': 2
    }
}

cfg.loss = {
    'name': 'CrossEntropyLoss',
    'kwargs': {

    },
    'multilabel': False

}

cfg.optimizer = {
    'name': 'Adam',
    'kwargs': {
        'lr': 3e-4
    }
}

cfg.lr_scheduler = {
    'name': 'StepLR',
    'kwargs': {
        'step_size': 5,
        'gamma': 1e-1
    },
    'options': {
        'interval': 'epoch'
    }
}

cfg.train = {
    'dataset': {
        'type': 'csv',
        'kwargs': {
            'csv_path': r'C:\Users\Sanek\dataset\cat_dog\train.csv',
            'image_prefix': r'C:\Users\Sanek\dataset\cat_dog\train'
        }
    },
    'dataloader': {
        'batch_size': 8,
        'num_workers': 0,
        'shuffle': True
    },
    'metrics': {
        'Precision': {
            'threshold': 0.5,
            'num_classes': 2
        }
    },
    'transforms': [
        {
            'name': 'HorizontalFlip',
            'kwargs': {
                'p': 0.5
            }
        },
        {
            'name': 'Resize',
            'kwargs': {
                'height': 224,
                'width': 224
            }
        },
        {
            'name': 'Normalize',
            'kwargs': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        {
            'name': 'ToTensorV2',
            'kwargs': {}
        },
    ]
}

cfg.val = {
    'dataset': {
        'type': 'csv',
        'kwargs': {
            'csv_path': r'C:\Users\Sanek\dataset\cat_dog\val.csv',
            'image_prefix': r'C:\Users\Sanek\dataset\cat_dog\val'
        }
    },
    'dataloader': {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': False
    },
    'metrics': {
        'Precision': {
            'threshold': 0.5,
            'num_classes': 2
        }
    },
    'transforms': [
        {
            'name': 'Normalize',
            'kwargs': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        {
            'name': 'ToTensorV2',
            'kwargs': {}
        },
    ]
}
