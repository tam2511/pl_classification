from pytorch_lightning.callbacks import Callback
import numpy as np
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader


class Mixup(Callback):
    def __init__(self, mode='batch', alpha=0.4):
        self.mode = mode
        self.alpha = alpha

    def __generate_dataset_sample(self, batch_size, dataloader):
        dataset = dataloader.dataset.datasets
        sample = [dataset[i] for i in np.random.randint(0, len(dataset) - 1, batch_size)]
        sample = dataloader.loaders.collate_fn(sample)
        sample_x, sample_y = sample
        return sample_x, sample_y

    def __generate_batch_sample(self, batch_x, batch_y):
        batch_size = len(batch_x)
        idxs = [np.random.choice([_ for _ in range(batch_size) if _ != i]) for i in range(batch_size)]
        idxs = torch.tensor(idxs, dtype=torch.long)
        sample_x, sample_y = batch_x[idxs], batch_y[idxs]
        return sample_x, sample_y

    def __unsqueeze(self, tensor, ndims=0, dim=0):
        for _ in range(ndims):
            tensor = tensor.unsqueeze(dim)
        return tensor

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        batch_x, batch_y = batch
        assert isinstance(batch_x, torch.Tensor)
        assert isinstance(batch_y, torch.Tensor)
        batch_size = batch_x.size(0)
        if self.mode == 'batch':
            batch_x_, batch_y_ = self.__generate_batch_sample(batch_x, batch_y)
        else:
            batch_x_, batch_y_ = self.__generate_dataset_sample(batch_size, trainer.train_dataloader)
        alpha = torch.from_numpy(np.random.beta(self.alpha, self.alpha, batch_size))
        batch_x_ = batch_x * self.__unsqueeze(alpha, 3, -1) + batch_x_ * self.__unsqueeze((1 - alpha), 3, -1)
        batch_y_ = batch_y * self.__unsqueeze(alpha, 1, -1) + batch_y_ * self.__unsqueeze((1 - alpha), 1, -1)
        for i in range(batch_size):
            batch_x[i] = batch_x_[i]
            batch_y[i] = batch_y_[i]
