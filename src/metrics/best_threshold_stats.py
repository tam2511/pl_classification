from torchmetrics import Metric
from torchmetrics.functional import stat_scores
from torch.nn import ModuleDict
import torch


class BestThresholdStats(Metric):
    def __init__(self, threshold_bins=0.01, min_threshold=0.0, max_threshold=1.0, decisive_metric_name='f1',
                 decisive_metric_kwargs={}, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.thresholds = torch.arange(min_threshold, max_threshold, threshold_bins)
        self.decisive_metric_name = decisive_metric_name
        self.decisive_metric_kwargs = decisive_metric_kwargs
        self.add_state("tp", default=torch.zeros_like(self.thresholds), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros_like(self.thresholds), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros_like(self.thresholds), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros_like(self.thresholds), dist_reduce_fx="sum")

    def compute(self):
        ...

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        stat_scores()

    def reset(self):
        ...
