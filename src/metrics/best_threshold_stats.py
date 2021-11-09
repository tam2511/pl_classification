from torchmetrics import Metric
from torchmetrics.functional import stat_scores
import torch


class BestThresholdStats(Metric):
    '''
    Metric with evaluating best threshold and compute precision, recall, accuracy, fbeta
    '''

    def __init__(
            self,
            num_classes,
            reduce='micro',
            threshold_bins=0.01,
            min_threshold=0.0,
            max_threshold=1.0,
            decisive_beta=1.0,
            epsilon=1e-8,
            dist_sync_on_step=False
    ):
        '''
        :param num_classes: number of classes
        :param reduce: 'micro' or 'macro'
        :param threshold_bins: step of threshold bins
        :param min_threshold: min threshold, which eval
        :param max_threshold: max threshold, which eval
        :param decisive_beta: beta for fbeta metric
        :param epsilon: epsilon for zero decision
        :param dist_sync_on_step: if true will be eval in step
        '''
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.thresholds = torch.arange(min_threshold, max_threshold, threshold_bins)
        self.decisive_alpha = decisive_beta
        self.num_classes = num_classes
        self.reduce = reduce
        self.epsilon = epsilon
        self.add_state("tp", default=torch.zeros(num_classes, self.thresholds.size(0)), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_classes, self.thresholds.size(0)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, self.thresholds.size(0)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, self.thresholds.size(0)), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for threshold_idx, threshold in enumerate(self.thresholds):
            stat_scores_ = stat_scores(preds, target, reduce='macro', num_classes=self.num_classes, threshold=threshold)
            self.tp[:, threshold_idx] += stat_scores_[:, 0]
            self.fp[:, threshold_idx] += stat_scores_[:, 1]
            self.tn[:, threshold_idx] += stat_scores_[:, 2]
            self.fn[:, threshold_idx] += stat_scores_[:, 3]

    def compute(self):
        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.epsilon)
        decision_metric_values = (1 + self.decisive_alpha ** 2) * (precision * recall) / (
                self.decisive_alpha ** 2 * precision + recall + self.epsilon)
        best_decision_metric_values, best_threshold_idx = torch.max(decision_metric_values, dim=1)
        best_thresholds = self.thresholds[best_threshold_idx]
        best_threshold_idx = best_threshold_idx.unsqueeze(1)
        precision, recall = precision.gather(1, best_threshold_idx)[:, 0], recall.gather(1, best_threshold_idx)[:, 0]
        accuracy = accuracy.gather(1, best_threshold_idx)[:, 0]
        if self.reduce == 'micro':
            precision, recall, accuracy = precision.mean(), recall.mean(), accuracy.mean()
            best_decision_metric_values = best_decision_metric_values.mean()
        return best_thresholds, best_decision_metric_values, precision, recall, accuracy

    def reset(self):
        self.tp *= 0
        self.tn *= 0
        self.fp *= 0
        self.fn *= 0
