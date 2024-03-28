from abc import abstractmethod

class BaseMethod:
    def __init__(self, loader, epochs, r=0.5, device="cuda"):
        self.r      = r
        self.device = device
        self.epochs = epochs
        self.loader = loader

    @abstractmethod
    def __call__(self, model, log_wandb=True):
        pass