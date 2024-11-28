import os
import tarfile
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image
from google.cloud import storage
import io
import random
import shutil
from simclr.modules.transformations.simclr import *


class ImageMaskDataset(VisionDataset):
    def __init__(self, bucket_name, train=True, split_percentage=0.7, validation_split_percentage=0.2, unlabeled_split_percentage=0.5, seed=42, transform=None, unlabeled=False, test=False, image_size=224, tar_prefix='data_tar/fish_data.tar.gz', download=False):
        super(ImageMaskDataset, self).__init__(root='data_tar/', transform=transform)
        
        self.train = train
        self.unlabeled = unlabeled  # Flag for unlabeled set
        self.test = test  # Flag for test set
        self.download = download
        self.tar_prefix = tar_prefix
        self.local_tar_path = os.path.join(self.root, 'fish_data.tar.gz')
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        
        # If download option is set or local data does not exist, download and extract the tar file
        if self.download or not self._local_data_exists():
            self._download_and_extract_tar()
        
        # Use local data after extraction
        self.image_paths = self._get_local_image_paths()
        
        # Debug: Print the number of images found
        print(f"Number of images found: {len(self.image_paths)}")
        if not self.image_paths:
            raise ValueError("No images found in the local 'data/' directory. Please check your tar file and extraction process.")

        # Get unique class names based on directory names
        self.classes = sorted(set(self._extract_class_name_from_path(path) for path in self.image_paths))
        
        # Create a class-to-index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        print(f"Class to index mapping: {self.class_to_idx}")
        
        # Assign labels or set to -1 if unlabeled
        self.labels = [-1] * len(self.image_paths) if self.unlabeled else [
            self.class_to_idx[self._extract_class_name_from_path(path)] for path in self.image_paths
        ]
        
        # Split data into unlabeled, train, validation, and test sets
        self._split_data(split_percentage, validation_split_percentage, unlabeled_split_percentage, seed)

        # Define a default transform if none is provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def _local_data_exists(self):
        if not os.path.exists(self.root):
            return False
        for class_name in os.listdir(self.root):
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path) and any(fname.endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(class_path)):
                return True
        return False
        
    def _download_and_extract_tar(self):
        self.local_tar_path = os.path.join('data_tar', 'fish_data.tar.gz')
        self.root = 'data_tar/'  # Set the root for extracted data

        os.makedirs('data_tar', exist_ok=True)

        if os.path.exists(self.local_tar_path):
            print(f"Tar file already exists locally at {self.local_tar_path}. Skipping download.")
        else:
            print(f"Downloading {self.tar_prefix} from GCS to {self.local_tar_path}...")
            blob = self.bucket.blob(self.tar_prefix)
            blob.download_to_filename(self.local_tar_path)
            print(f"Downloaded {self.tar_prefix} to {self.local_tar_path}")

        print(f"Extracting {self.local_tar_path}...")
        with tarfile.open(self.local_tar_path, 'r:gz') as tar:
            tar.extractall(path=self.root)

    def _get_local_image_paths(self):
        local_image_paths = []
        print("Collecting image paths from the 'data_tar/' directory...")
        for root, dirs, files in os.walk(self.root):
            if "GT" in root:
                continue

            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file_name)
                    local_image_paths.append(image_path)
                   # print(f"Found image: {image_path}")

       # print(f"Found {len(local_image_paths)} images excluding GT directories.")
        return local_image_paths

    def _extract_class_name_from_path(self, path):
        return os.path.basename(os.path.dirname(path))

    def _split_data(self, split_percentage, validation_split_percentage, unlabeled_split_percentage, seed):
        total_size = len(self.image_paths)
        if total_size == 0:
            raise ValueError("Dataset has no images. Check your local 'data/' directory.")

        indices = list(range(total_size))
        random.seed(seed)
        random.shuffle(indices)
        
        unlabeled_size = int(total_size * unlabeled_split_percentage)
        remaining_size = total_size - unlabeled_size
        train_size = int(remaining_size * split_percentage)
        validation_size = int(remaining_size * validation_split_percentage)
        test_size = remaining_size - train_size - validation_size

        self.unlabeled_indices = indices[:unlabeled_size]
        remaining_indices = indices[unlabeled_size:]
        self.train_indices = remaining_indices[:train_size]
        self.val_indices = remaining_indices[train_size:train_size + validation_size]
        self.test_indices = remaining_indices[train_size + validation_size:]
        
        if self.unlabeled:
            self.indices = self.unlabeled_indices
            print(f"Unlabeled dataset with {len(self.indices)} samples.")
        elif self.train:
            self.indices = self.train_indices
            print(f"Training dataset with {len(self.indices)} samples.")
        elif self.test:
            self.indices = self.test_indices
            print(f"Test dataset with {len(self.indices)} samples.")
        else:
            self.indices = self.val_indices
            print(f"Validation dataset with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        label = self.labels[real_idx]

        image = Image.open(img_path).convert('RGB')
        if self.unlabeled and callable(self.transform):
            image1, image2 = self.transform(image)
            return (image1, image2), label
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    size = 224
    transform = TransformsSimCLR(size=size)
    #transform = None
    download = True
    
    bucket_name = "fish-dataset-cl"  # Replace with your bucket name

    # Unlabeled dataset for pretraining
    unlabeled_dataset = ImageMaskDataset(bucket_name, transform=transform, unlabeled=True, download=download)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)
    #print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")

    # Train dataset for fine-tuning
    train_dataset = ImageMaskDataset(bucket_name, train=True, transform=transform.test_transform, unlabeled=False, download=download)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    #print(f"Train dataset size: {len(train_dataset)}")

    # Validation dataset for fine-tuning
    val_dataset = ImageMaskDataset(bucket_name, train=False, transform=transform.test_transform, unlabeled=False, download=download)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
#    print(f"Validation dataset size: {len(val_dataset)}")

    # Test dataset for fine-tuning
    test_dataset = ImageMaskDataset(bucket_name, train=False, transform=transform.test_transform, unlabeled=False, test=True, download=download)
    test_loader = torch.utils.data.DataLoader
    
    import ipdb;ipdb.set_trace()
