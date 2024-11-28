
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from simclr.modules.dataloader_transform import ImageMaskDataset
from simclr.args import get_args
import os
from simclr.modules.transformations import TransformsSimCLR


args = get_args()

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4, path='checkpoint.pth', verbose=False):
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


# Define the model
class ResNet50LogisticRegression(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50LogisticRegression, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        return self.resnet50(x)

# Helper function to save the model
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

# Helper function to load the model
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

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

# Evaluation function
def evaluate(model, loader, criterion, device, split_name="Validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating {split_name}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"{split_name} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# Main training loop with model saving
def main_training_loop(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, checkpoint_path):
    best_loss = float('inf')
    start_epoch = 0

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, delta=1e-4, path=checkpoint_path, verbose=True)

    # Load checkpoint if available
    model, optimizer, scheduler, start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Training phase
        train_loss = train(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, split_name="Validation")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss, model, optimizer, scheduler, epoch)

        # Terminate if early stopping criterion is met
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        print(f"Epoch {epoch + 1} completed. Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Load the best model from early stopping checkpoint
    model, optimizer, scheduler, _, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    # Test phase after training completes
    print("Evaluating on Test Set:")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, split_name="Test")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")



if __name__ == '__main__':
    
    train_dataset = ImageMaskDataset(
                    bucket_name = args.bucket_name,
                    image_size=args.image_size,
                    train=True,
                    unlabeled=False,
                    unlabeled_split_percentage=0.9,
                    transform = TransformsSimCLR(size=args.image_size).test_transform)

    val_dataset = ImageMaskDataset(
                    bucket_name = args.bucket_name,
                    image_size=args.image_size,
                    unlabeled=False,
                    train=False,
                    test=False,
                    unlabeled_split_percentage=0.9,
                    transform = TransformsSimCLR(size=args.image_size).test_transform)

    test_dataset = ImageMaskDataset(
                    bucket_name = args.bucket_name,
                    image_size=args.image_size,
                    unlabeled=False,
                    train=False,
                    test=True,
                    unlabeled_split_percentage=0.9,
                    transform = TransformsSimCLR(size=args.image_size).test_transform)
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
        )

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
        )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
        )
    
    # Initialize the model
    num_classes = len(train_dataset.classes)
    model = ResNet50LogisticRegression(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.resnet50.fc.parameters(), lr=1e-4)

    # Define the ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Use 'min' since we're monitoring the validation loss
        factor=0.1,        # Reduce the LR by a factor of 0.1
        patience=5,        # Wait for 5 epochs of no improvement
        threshold=1e-4,    # Threshold for measuring the improvement
        min_lr=1e-6,       # Minimum learning rate
        verbose=True       # Print a message when LR is reduced
    )

    # Define the number of epochs and checkpoint path
    num_epochs = 100
    checkpoint_path = "resnet_finetuned.pth"

    # Start the main training loop with train, validation, and test loaders
    main_training_loop(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        checkpoint_path
    )

