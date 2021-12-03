from pytorch_lightning import LightningModule
import torch

from lr_schedulers.builder import create_lr_scheduler
from metrics.metric import MetricsList
from optimizers.builder import create_optimizer


class TripletsLearner(LightningModule):
    def __init__(
            self,
            cfg,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            train_metrics: list = None,
            val_metrics: list = None,
            return_val_output=False,
            return_train_output=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_f = loss
        self.train_metrics = MetricsList()
        self.return_val_output = return_val_output
        self.return_train_output = return_train_output
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
        data, anchor_labels = batch
        anchors, positives, negatives = data
        anchors, positives, negatives = self.forward(anchors), self.forward(positives), self.forward(negatives)
        loss = self.loss_f(anchors, positives, negatives)
        self.log('train/loss', loss, on_step=True, on_epoch=False)
        self.train_metrics.update(anchors, anchor_labels)
        ret = {'loss': loss}
        if self.return_train_output:
            ret['output'] = (anchors, anchor_labels)
        return ret

    def training_epoch_end(self, train_step_outputs):
        train_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        for metric_name in train_metrics:
            self.log(f'train/{metric_name}', train_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        data, anchor_labels = batch
        anchors = data[0]
        anchors = self.forward(anchors)
        self.val_metrics.update(anchors, anchor_labels)
        ret = {}
        if self.return_val_output:
            ret['output'] = (anchors, anchor_labels)
        return ret

    def validation_epoch_end(self, val_step_outputs):
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        for metric_name in val_metrics:
            self.log(f'val/{metric_name}', val_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        '''TODO: user's optimizer and lr_scheduler'''
        optimizer = create_optimizer(self.cfg.optimizer.name, self.model, self.cfg.optimizer.kwargs)
        lr_scheduler = create_lr_scheduler(self.cfg.lr_scheduler.name, optimizer, self.cfg.lr_scheduler.kwargs)
        lr_scheduler_options = self.cfg.lr_scheduler.options
        lr_scheduler_options['scheduler'] = lr_scheduler
        return [optimizer], [lr_scheduler_options]
