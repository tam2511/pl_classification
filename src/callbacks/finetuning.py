from pytorch_lightning.callbacks import BaseFinetuning


class SerialFinetune(BaseFinetuning):
    # TODO: serial finetuner
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        ...

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        ...
