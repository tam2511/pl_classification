from torchmetrics import *
import torch

available_names = [
    'Accuracy', 'Precision', 'Recall', 'AveragePrecision'
]


class TorchMetric(Metric):
    def __init__(self, metric_info, class_names=None):
        super().__init__()
        if metric_info['name'] in available_names:
            self.metric = eval(metric_info['name'])(**{_: metric_info[_] for _ in metric_info if _ != 'name'})
        else:
            raise NotImplementedError('{} not implemented'.format(metric_info['name']))
        self.class_names = class_names
        self.metric_info = metric_info

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.metric.update(preds, target)

    def compute(self) -> dict:
        result = self.metric.compute()
        if result.size(0) == len(self.class_names):
            return {'{}_{}'.format(self.metric_info['name'], self.class_names[idx]): result[idx] for idx in
                    range(len(self.class_names))}
        return {'{}_{}'.format(self.metric_info['name'], self.metric.average): result}

    def reset(self):
        self.metric.reset()
