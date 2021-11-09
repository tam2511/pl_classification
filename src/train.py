from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from datasets.builder import create_dataset
from datasets.transforms import create_transform
from learners.classificator_learner import ClassificatorLearner
from configs.resnet18_example import cfg

train_transform = create_transform(cfg.train.transforms)
val_transforms = create_transform(cfg.val.transforms)

if __name__ == '__main__':
    train_dataset = create_dataset(cfg.train.dataset.type, train_transform, cfg.train.dataset.kwargs)
    val_dataset = create_dataset(cfg.val.dataset.type, val_transforms, cfg.val.dataset.kwargs)

    train_dataloader = DataLoader(dataset=train_dataset, **cfg.train.dataloader)
    val_dataloader = DataLoader(dataset=val_dataset, **cfg.val.dataloader)

    learner = ClassificatorLearner(cfg=cfg)
    trainer = Trainer(**cfg.trainer_kwargs)
    trainer.fit(learner, train_dataloaders=[train_dataloader], val_dataloaders=[val_dataloader])
