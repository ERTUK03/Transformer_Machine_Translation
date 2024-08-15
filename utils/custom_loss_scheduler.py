import torch

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = max(1, self._step_count)
        scale = self.d_model ** -0.5
        lr = scale * min(step_num ** -0.5, step_num * (self.warmup_steps ** -1.5))
        return [lr for _ in self.base_lrs]
