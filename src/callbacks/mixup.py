from pytorch_lightning.callbacks import Callback


class Mixup(Callback):
    def __init__(self, mode='batch', alpha=0.4, p=0.1):
        self.mode = mode
        self.alpha = alpha
        self.p = p

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        ...
