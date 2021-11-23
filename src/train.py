from pytorch_lightning import Trainer
from callbacks import SequentialFinetune, ImageLogger, Mixup
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.builder import create_dataset
from datasets.transforms import create_transform
from learners.classificator_learner import ClassificatorLearner
from configs.resnet18_example import cfg
from models.builder import create_model
from losses.builder import create_loss

train_transform = create_transform(cfg.train.transforms)
val_transforms = create_transform(cfg.val.transforms)

finetuner = SequentialFinetune(cfg.finetuning)
image_logger = ImageLogger(mode='train', n_images=100, class_names=['cat', 'dog'], multilabel=True, n_top_classes=5)
mixup = Mixup(alpha=0.4, mode='dataset')

if __name__ == '__main__':
    train_dataset = create_dataset(cfg.train.dataset.type, train_transform, cfg.train.dataset.kwargs)
    val_dataset = create_dataset(cfg.val.dataset.type, val_transforms, cfg.val.dataset.kwargs)

    train_dataloader = DataLoader(dataset=train_dataset, **cfg.train.dataloader)
    val_dataloader = DataLoader(dataset=val_dataset, **cfg.val.dataloader)

    model = create_model(cfg.model.name, cfg.model.domen, cfg.model.kwargs)

    loss = create_loss(cfg.loss.name, cfg.multilabel, cfg.loss.kwargs)

    learner = ClassificatorLearner(cfg=cfg, model=model, loss=loss, return_val_output=True, return_train_output=True)
    trainer = Trainer(gpus=0, callbacks=[finetuner, image_logger, mixup])
    trainer.fit(learner, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader])
