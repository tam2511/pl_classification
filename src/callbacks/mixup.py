import numpy as np
import torch
from callbacks.mix_callback import MixBaseCallback


class Mixup(MixBaseCallback):
    def __init__(self, mode='batch', alpha=0.4):
        super().__init__()
        self.mode = mode
        self.alpha = alpha

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        batch_x, batch_y = batch
        assert isinstance(batch_x, torch.Tensor)
        assert isinstance(batch_y, torch.Tensor)
        batch_size = batch_x.size(0)
        if self.mode == 'batch':
            batch_x_, batch_y_ = self._generate_batch_sample(batch_x, batch_y)
        else:
            batch_x_, batch_y_ = self._generate_dataset_sample(batch_size, trainer.train_dataloader)
        alpha = torch.from_numpy(np.random.beta(self.alpha, self.alpha, batch_size))
        batch_x_ = batch_x * self._unsqueeze(alpha, 3, -1) + batch_x_ * self._unsqueeze((1 - alpha), 3, -1)
        batch_y_ = batch_y * self._unsqueeze(alpha, 1, -1) + batch_y_ * self._unsqueeze((1 - alpha), 1, -1)
        for i in range(batch_size):
            batch_x[i] = batch_x_[i]
            batch_y[i] = batch_y_[i]
