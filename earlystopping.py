import numpy as np
import torch
import datetime

from config import *

class EarlyStopping:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Model Save
        nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
        PATH = "./Classification_" + nowDate + ".pth"
        torch.save({
            'epoch': NUM_EPOCH,
            'model_state_dict': model.model.state_dict(),
            'loss': val_loss,
        }, PATH)

        self.val_loss_min = val_loss