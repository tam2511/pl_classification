import torch

from learners.base_learner import BaseLearner


class TripletsLearner(BaseLearner):
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
        data, anchor_labels = batch
        if not isinstance(data, list):
            raise ValueError('Dataset must return tuple or list if images.')
        if len(data) == 3:
            anchors, positives, negatives = data
            anchors, positives, negatives = self.forward(anchors), self.forward(positives), self.forward(negatives)
            loss = self.loss_f(anchors, positives, negatives)
        elif len(data) == 1:
            anchors = data[0]
            anchors = self.forward(anchors)
            loss = torch.tensor(0.0)
        else:
            raise ValueError('Dataset can return 1- or 3-sized list of images')
        output = anchors
        target = anchor_labels
        return_output = anchors
        return loss, output, target, return_output
