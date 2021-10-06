from torchmetrics.classification import Accuracy, Precision, Recall, AveragePrecision
from torchmetrics import Metric
from torch.nn import ModuleDict
import torch


def create_metrics(metrics_infos: dict) -> Metric:
    '''
    Create Metric object from dict of metrics and info
    :param metrics_infos: dict as like {metrics_name: metric_kwargs}
    :return: Metric
    '''
    return MetricDict(list(metrics_infos.keys()), list(metrics_infos.values()))


def create_metric__(metric_name, kwargs):
    if metric_name == 'Accuracy':
        return Accuracy(**kwargs)
    elif metric_name == 'Precision':
        return Precision(**kwargs)
    elif metric_name == 'Recall':
        return Recall(**kwargs)
    elif metric_name == 'AveragePrecision':
        return AveragePrecision(**kwargs)
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
