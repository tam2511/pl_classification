import torch

from learners.base_learner import BaseLearner


class ClassificatorLearner(BaseLearner):
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
        super().__init__(cfg, model, loss, train_metrics, val_metrics, return_val_output, return_train_output)

    __init__.__doc__ = BaseLearner.__init__.__doc__

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_f(y_, y.float() if self.cfg.multilabel else y.argmax(dim=1))
        output = y_.sigmoid() if self.cfg.multilabel else y_.argmax(dim=1)
        target = y
        return_output = y_.sigmoid() if self.cfg.multilabel else y_.softmax(dim=1)
        return loss, output, target, return_output
