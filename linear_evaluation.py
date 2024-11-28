import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.dataloader_transform import ImageMaskDataset
from simclr.modules.early_stopping import EarlyStopping
from simclr.modules.utils import check_duplicates

from model import load_optimizer

from utils import yaml_config_hook

import sys

def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, val_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    
    if val_loader is not None:
        val_X, val_y = inference(val_loader, simclr_model, device)

        return train_X, train_y, test_X, test_y, val_X, val_y
    else:   
        return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, X_val, y_val, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    
    if X_val is not None:
        val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader, val_loader
    
    else:
        return train_loader, test_loader
    
def train_one_epoch(args, train_loader, model, criterion, optimizer):
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    # Average metrics
    loss_epoch /= len(train_loader)
    accuracy_epoch /= len(train_loader)

    return loss_epoch, accuracy_epoch


def train(args, train_loader, val_loader, model, criterion, optimizer, scheduler):
    for epoch in range(args.logistic_epochs):
        model.train()
        loss_epoch = 0
        accuracy_epoch = 0

        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = criterion(output, y)

            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        # Average metrics over batches
        loss_epoch /= len(train_loader)
        accuracy_epoch /= len(train_loader)

        # Validation step
        val_loss, val_accuracy = validate(val_loader, model, criterion, args.device)

        # Step the scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{args.logistic_epochs}]\t"
            f"Train Loss: {loss_epoch:.4f}\t Train Accuracy: {accuracy_epoch:.4f}\t"
            f"Val Loss: {val_loss:.4f}\t Val Accuracy: {val_accuracy:.4f}\t"
            f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}"
        )

def validate(loader, model, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)

            val_loss += loss.item()
            val_accuracy += acc

    val_loss /= len(loader)
    val_accuracy /= len(loader)
    return val_loss, val_accuracy

def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="SimCLR")
        config = yaml_config_hook("./config/config.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        if args.dataset == "STL10":
            train_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="test",
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=False,
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == 'FISH':
            train_dataset = ImageMaskDataset(
                    bucket_name = args.bucket_name,
                    image_size=args.image_size,
                    train=True,
                    unlabeled=False,
                    unlabeled_split_percentage=0.90,
                    transform = TransformsSimCLR(size=args.image_size).test_transform)

            val_dataset = ImageMaskDataset(
                    bucket_name = args.bucket_name,
                    image_size=args.image_size,
                    unlabeled=False,
                    train=False,
                    test=False,
                    unlabeled_split_percentage=0.90,
                    transform = TransformsSimCLR(size=args.image_size).test_transform)

            test_dataset = ImageMaskDataset(
                    bucket_name = args.bucket_name,
                    image_size=args.image_size,
                    unlabeled=False,
                    train=False,
                    test=True,
                    unlabeled_split_percentage=0.90,
                    transform = TransformsSimCLR(size=args.image_size).test_transform)
        else:
            raise NotImplementedError


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

        encoder = get_resnet(args.resnet, pretrained=True)
        n_features = encoder.fc.in_features  # get dimensions of fc layer

        # load pre-trained model from checkpoint
        simclr_model = SimCLR(encoder, args.projection_dim, n_features)
        #model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        model_fp = 'models/contrastive_epoch50_finetuned.pth'
        simclr_model = torch.load(model_fp, map_location=args.device.type)
        #simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        simclr_model = simclr_model.to(args.device)
        simclr_model.eval()

        ## Logistic Regression
        n_classes = 9  # FISH
        model = LogisticRegression(simclr_model.n_features, n_classes)
        model = model.to(args.device)

        #optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        optimizer, scheduler = load_optimizer(args, model)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"### Creating features for {args.dataset} from pre-trained context model ###")
        if args.dataset == 'STL10' or args.dataset == 'CIFAR10':
            (train_X, train_y, test_X, test_y) = get_features(
                simclr_model=simclr_model, train_loader=train_loader, test_loader=test_loader, device=args.device
            )

            arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
                X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y, batch_size= args.logistic_batch_size
            )

            for epoch in range(args.logistic_epochs):
                loss_epoch, accuracy_epoch = train(
                    args, arr_train_loader, simclr_model, model, criterion, optimizer
                )
                print(
                    f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
                )

            # final testing
            loss_epoch, accuracy_epoch = test(
                args, arr_test_loader, simclr_model, model, criterion, optimizer
            )
            print(
                f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
            )

        elif args.dataset == 'FISH':
            (train_X, train_y, test_X, test_y, val_X, val_y) = get_features(
                simclr_model=simclr_model, train_loader=train_loader, test_loader=test_loader, 
                val_loader=val_loader, device=args.device)

            arr_train_loader, arr_test_loader, arr_val_loader = create_data_loaders_from_arrays(
                X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y, 
                X_val=val_X, y_val=val_y, batch_size=args.logistic_batch_size
            )
            
            if (
                check_duplicates(train_X, test_X) or
                check_duplicates(val_X, test_X) or
                check_duplicates(train_X, val_X)
                    ):
                print("Terminating script due to duplicates.")
                sys.exit(1)  # Exit script with an error code
            else:
                print("No duplicates found. Proceeding with the script...")
                pass
            
            # EarlyStopping and best_val_loss initialization
        early_stopping = EarlyStopping(patience=args.logistic_patience, verbose=True)
        best_val_loss = float('inf')

        # Main epoch loop
        for epoch in range(args.logistic_epochs):
            print(f"Epoch {epoch+1}/{args.logistic_epochs}")

            # Training step
            train_loss, train_accuracy = train_one_epoch(args, arr_train_loader, model, criterion, optimizer)

            # Validation step
            val_loss, val_accuracy = validate(arr_val_loader, model, criterion, args.device)

            # Check early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Model saved at epoch {epoch+1}")
                torch.save(model.state_dict(), "logistic_finetuned.pth")

            # Logging
            print(
                f"Epoch [{epoch+1}/{args.logistic_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\t"
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

        # Load the best model for final testing
        model.load_state_dict(torch.load("logistic_finetuned.pth"))

        # Final testing
        loss_epoch, accuracy_epoch = test(
            args, arr_test_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader):.4f}\t Accuracy: {accuracy_epoch / len(arr_test_loader):.4f}"
        )
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)     
