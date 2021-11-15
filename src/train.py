from pytorch_lightning import Trainer
from callbacks import SequentialFinetune
from torch.utils.data import DataLoader

from datasets.builder import create_dataset
from datasets.transforms import create_transform
from learners.classificator_learner import ClassificatorLearner
from configs.resnet18_example import cfg
from models.builder import create_model
from losses.builder import create_loss

train_transform = create_transform(cfg.train.transforms)
val_transforms = create_transform(cfg.val.transforms)

finetuner = SequentialFinetune(cfg.finetuning)

if __name__ == '__main__':
    train_dataset = create_dataset(cfg.train.dataset.type, train_transform, cfg.train.dataset.kwargs)
    val_dataset = create_dataset(cfg.val.dataset.type, val_transforms, cfg.val.dataset.kwargs)

    train_dataloader = DataLoader(dataset=train_dataset, **cfg.train.dataloader)
    val_dataloader = DataLoader(dataset=val_dataset, **cfg.val.dataloader)

    model = create_model(cfg.model.name, cfg.model.domen, cfg.model.kwargs)

    loss = create_loss(cfg.loss.name, cfg.multilabel, cfg.loss.kwargs)

    learner = ClassificatorLearner(cfg=cfg, model=model, loss=loss)
    trainer = Trainer(gpus=0, callbacks=[finetuner])
    trainer.fit(learner, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader])
