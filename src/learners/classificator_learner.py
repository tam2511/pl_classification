from pytorch_lightning import LightningModule
from torch.nn.functional import sigmoid

from losses.builder import create_loss
from lr_schedulers.builder import create_lr_scheduler
from metrics.builder import create_metrics
from models.builder import create_model
from optimizers.builder import create_optimizer


class ClassificatorLearner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(cfg.model.name, cfg.model.domen, cfg.model.kwargs)
        self.loss_f = create_loss(cfg.loss.name, cfg.loss.multilabel, cfg.loss.kwargs)
        self.train_metrics = create_metrics(cfg.train.metrics)
        self.val_metrics = create_metrics(cfg.val.metrics)

    def forward(self, x):
        return sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_f(y_, y if self.cfg.loss.multilabel else y.argmax(dim=1))
        self.log('train/loss', loss, on_step=True, on_epoch=False)
        self.train_metrics.update(y_, y)
        return loss

    def training_epoch_end(self, train_step_outputs):
        train_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        for metric_name in train_metrics:
            self.log(f'train/{metric_name}', train_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_f(y_, y if self.cfg.loss.multilabel else y.argmax(dim=1))
        self.val_metrics.update(y_, y)
        self.log(f'val/loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        for metric_name in val_metrics:
            self.log(f'val/{metric_name}', val_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer.name, self.model, self.cfg.optimizer.kwargs)
        lr_scheduler = create_lr_scheduler(self.cfg.lr_scheduler.name, optimizer, self.cfg.lr_scheduler.kwargs)
        lr_scheduler_options = self.cfg.lr_scheduler.options
        lr_scheduler_options['scheduler'] = lr_scheduler
        return [optimizer], [lr_scheduler_options]
