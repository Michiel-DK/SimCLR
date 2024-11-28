class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.00001, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait after no improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Print messages about early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
