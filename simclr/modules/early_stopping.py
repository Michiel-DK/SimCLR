
                
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        # Check if val_loss improved significantly
        if val_loss < self.best_loss - self.min_delta:
            # if self.verbose:
            #     print(f"Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}. Resetting counter.")
            self.best_loss = val_loss
            self.counter = 0  # Reset counter
        else:
            self.counter += 1  # Increment counter
            if self.verbose:
                print(f"No improvement in validation loss: {val_loss:.4f}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

