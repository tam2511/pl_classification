from torchmetrics.classification import *
from torchmetrics import Metric
from torch.nn import ModuleDict
import torch

available_names = [
    'Accuracy', 'Precision', 'Recall', 'AveragePrecision'
]


def create_metrics(metrics_infos: dict) -> Metric:
    '''
    Create Metric object from dict of metrics and info
    :param metrics_infos: dict as like {metrics_name: metric_kwargs}
    :return: Metric
    '''
    return MetricDict(list(metrics_infos.keys()), metrics_infos)


def create_metric__(metric_name, kwargs):
    if metric_name in available_names:
        return eval(metric_name)(**kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(metric_name))


class MetricDict(Metric):
    def __init__(self, metrics_names, metrics_kwargs, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.metrics = ModuleDict()
        for metric_name in metrics_names:
            metric = create_metric__(metric_name, metrics_kwargs[metric_name])
            self.metrics.add_module(name=metric_name, module=metric)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        for metric_name in self.metrics.keys():
            self.metrics[metric_name].to(preds.device)
            self.metrics[metric_name].update(preds, target)

    def compute(self):
        return {
            metric_name: self.metrics[metric_name].compute() for metric_name in self.metrics.keys()
        }

    def reset(self):
        for metric_name in self.metrics.keys():
            self.metrics[metric_name].reset()

    def add(self, metric: Metric, name: str):
        self.metrics.add_module(name=name, module=metric)
