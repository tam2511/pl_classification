from torchmetrics import Metric
from torch.nn import ModuleList
import torch
from metrics.metric import TorchMetric
from metrics.best_threshold_stats import BestThresholdStats


def create_metrics(metrics_infos: list, class_names: list = None) -> Metric:
    '''
    Create Metric object from dict of metrics and info
    :param metrics_infos: list of dicts
    :return: Metric
    '''
    return MetricsList(metrics_infos, class_names)


def create_metric__(metric_info, class_names=None):
    if 'name' not in metric_info:
        raise ValueError('Metric with info {} has not name key'.format(metric_info))
    if metric_info['name'] == 'BestThresholdStats':
        return BestThresholdStats(metric_info, class_names)
    return TorchMetric(metric_info, class_names)


class MetricsList(Metric):
    def __init__(self, metrics_infos, class_names=None, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.metrics = ModuleList()
        for idx, metric_info in enumerate(metrics_infos):
            metric = create_metric__(metric_info, class_names)
            self.metrics.append(module=metric)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        for metric_idx in range(len(self.metrics)):
            # TODO: optimize metrics device passing
            self.metrics[metric_idx].to(preds.device)
            self.metrics[metric_idx].update(preds, target)

    def compute(self):
        result = []
        for metric_idx in range(len(self.metrics)):
            result_ = self.metrics[metric_idx].compute()
            for name in result_:
                result[name] = result_[name]
        return result

    def reset(self):
        for metric_idx in range(len(self.metrics)):
            self.metrics[metric_idx].reset()

    def add(self, metric: Metric):
        self.metrics.append(module=metric)
