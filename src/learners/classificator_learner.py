import torch

from learners.base_learner import BaseLearner
from optimizers import Optimizer
from lr_schedulers import Scheduler


class ClassificatorLearner(BaseLearner):
    def __init__(
            self,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: Optimizer,
            multilabel: bool = False,
            lr_scheduler: Scheduler = None,
            train_metrics: list = None,
            val_metrics: list = None,
            return_val_output=False,
            return_train_output=False,
    ):
        super().__init__(model, loss, optimizer, lr_scheduler, train_metrics, val_metrics, return_val_output,
                         return_train_output)
        self.multilabel = multilabel

    __init__.__doc__ = BaseLearner.__init__.__doc__

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_f(y_, y.float() if self.multilabel else y.argmax(dim=1))
        output = y_.sigmoid() if self.multilabel else y_.argmax(dim=1)
        target = y
        return_output = y_.sigmoid() if self.multilabel else y_.softmax(dim=1)
        return loss, output, target, return_output
