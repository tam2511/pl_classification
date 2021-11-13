from pytorch_lightning import LightningModule
import torch

from losses.builder import create_loss
from lr_schedulers.builder import create_lr_scheduler
from metrics.metric import MetricsList
from models.builder import create_model
from optimizers.builder import create_optimizer


class ClassificatorLearner(LightningModule):
    def __init__(
            self,
            cfg,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            train_metrics: list = None,
            val_metrics: list = None
    ):
        # TODO: add metrics options
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_f = loss
        self.train_metrics = MetricsList()
        if not train_metrics is None:
            for train_metric in train_metrics:
                self.train_metrics.add(metric=train_metric)
        self.val_metrics = MetricsList([])
        if not val_metrics is None:
            for val_metric in val_metrics:
                self.val_metrics.add(metric=val_metric)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_f(y_, y.float() if self.cfg.loss.multilabel else y.argmax(dim=1))
        self.log('train/loss', loss, on_step=True, on_epoch=False)
        self.train_metrics.update(y_, y)
        return loss

    def training_epoch_end(self, train_step_outputs):
        train_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        for metric_name in train_metrics:
            in_progress_bar = metric_name in self.cfg.train.progress_bar
            self.log(f'train/{metric_name}', train_metrics[metric_name], on_step=False, on_epoch=True,
                     prog_bar=in_progress_bar)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_f(y_, y.float() if self.cfg.loss.multilabel else y.argmax(dim=1))
        self.val_metrics.update(y_, y)
        self.log(f'val/loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        for metric_name in val_metrics:
            metric_name_ = metric_name.split('_')[0]
            in_progress_bar = metric_name_ in self.cfg.val.progress_bar
            self.log(f'val/{metric_name}', val_metrics[metric_name], on_step=False, on_epoch=True,
                     prog_bar=in_progress_bar)

    def configure_optimizers(self):
        '''TODO: user's optimizer and lr_scheduler'''
        optimizer = create_optimizer(self.cfg.optimizer.name, self.model, self.cfg.optimizer.kwargs)
        lr_scheduler = create_lr_scheduler(self.cfg.lr_scheduler.name, optimizer, self.cfg.lr_scheduler.kwargs)
        lr_scheduler_options = self.cfg.lr_scheduler.options
        lr_scheduler_options['scheduler'] = lr_scheduler
        return [optimizer], [lr_scheduler_options]