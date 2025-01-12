import torch
import os

                
class EarlyStoppingSimCLR:
    def __init__(self, patience=5, min_delta=0.0001, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        #Check if val_loss improved significantly
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}. Resetting counter.")
            self.best_loss = val_loss
            self.counter = 0  #Reset counter
        else:
            self.counter += 1  #Increment counter
            if self.verbose:
                print(f"No improvement in validation loss: {val_loss:.4f}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3, path='checkpoint.pth', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model checkpoint.
            verbose (bool): Print messages for early stopping events.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        """
        Saves model when validation loss decreases.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")
            
#Helper function to save the model
def save_checkpoint(epoch, model, optimizer, scheduler, loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")

#Helper function to load the model
def load_checkpoint(file_path, model, optimizer, scheduler):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {file_path}, starting at epoch {start_epoch}")
        return model, optimizer, scheduler, start_epoch, loss
    else:
        print(f"No checkpoint found at {file_path}")
        return model, optimizer, scheduler, 0, float('inf')