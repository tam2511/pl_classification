from torchmetrics import Metric
from torchmetrics.functional import stat_scores
import torch


class BestThresholdStats_(Metric):
    '''
    Metric looks for the best threshold for the metric fbeta and calculates the precision, recall, accuracy and fbeta
    '''

    def __init__(
            self,
            num_classes: int,
            average: str = 'micro',
            threshold_bins: float = 0.01,
            min_threshold: float = 0.0,
            max_threshold: float = 1.0,
            decisive_beta: float = 1.0,
            epsilon: float = 1e-8,
            class_names: list = None,
            dist_sync_on_step: bool = False,
            compute_on_step: bool = True
    ):
        '''
        :param num_classes: number of classes
        :param average: 'micro' or 'macro' or 'none
        :param threshold_bins: step of threshold bins
        :param min_threshold: min threshold, which eval
        :param max_threshold: max threshold, which eval
        :param decisive_beta: beta for fbeta metric
        :param epsilon: epsilon for zero decision
        :param class_names: list of string names of classes
        '''
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.thresholds = torch.arange(min_threshold, max_threshold, threshold_bins)
        self.decisive_alpha = decisive_beta
        self.num_classes = num_classes
        self.class_names = class_names
        self.name = self.__class__.__name__
        if average not in ['micro', 'macro', 'none']:
            raise ValueError('available only micro, macro or none')
        self.average = average
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

    def __compute(self):
        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)
        decision_metric_values = (1 + self.decisive_alpha ** 2) * (precision * recall) / (
                self.decisive_alpha ** 2 * precision + recall + self.epsilon)
        best_decision_metric_values, best_threshold_idx = torch.max(decision_metric_values, dim=1)
        best_thresholds = self.thresholds[best_threshold_idx]
        best_threshold_idx = best_threshold_idx.unsqueeze(1)
        tp, fp, tn, fn = self.tp.gather(1, best_threshold_idx)[:, 0], self.fp.gather(1, best_threshold_idx)[:, 0], \
                         self.tn.gather(1, best_threshold_idx)[:, 0], self.fn.gather(1, best_threshold_idx)[:, 0]
        if self.average == 'micro':
            tp, fp, tn, fn = tp.sum(), fp.sum(), tn.sum(), fn.sum()
        precision, recall, accuracy = tp / (tp + fp + self.epsilon), tp / (tp + fn + self.epsilon), (tp + tn) / (
                tp + tn + fp + fn)
        if self.average == 'macro':
            precision, recall, accuracy = precision.mean(), recall.mean(), accuracy.mean()
        return best_thresholds, best_decision_metric_values, precision, recall, accuracy

    def compute(self) -> dict:
        best_thresholds, best_decision_metric_values, precision, recall, accuracy = self.__compute()
        result = {}
        if precision.size() == torch.Size([len(self.class_names)]):
            for idx in range(len(self.class_names)):
                result['{}_precision_{}'.format(self.name, self.class_names[idx])] = precision[idx]
                result['{}_recall_{}'.format(self.name, self.class_names[idx])] = recall[idx]
                result['{}_accuracy_{}'.format(self.name, self.class_names[idx])] = accuracy[idx]
        else:
            result['{}_precision_{}'.format(self.name, self.metric.average)] = precision
            result['{}_recall_{}'.format(self.name, self.metric.average)] = recall
            result['{}_accuracy_{}'.format(self.name, self.metric.average)] = accuracy
        for idx in range(len(self.class_names)):
            result['{}_f({:.2f})_{}'.format(self.name, self.metric.decisive_alpha,
                                            self.class_names[idx])] = best_decision_metric_values[idx]
            result['{}_threshold_{}'.format(self.name, self.class_names[idx])] = best_thresholds[idx]
        return result

    def reset(self):
        self.tp *= 0
        self.tn *= 0
        self.fp *= 0
        self.fn *= 0
