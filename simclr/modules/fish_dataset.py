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

class ImageMaskDataset(VisionDataset):
    def __init__(self, bucket_name, split_type='train', train_percentage=0.5, val_percentage=0.3, seed=42, transform=None, image_size=224, tar_prefix='data_tar/fish_data.tar.gz', download=False):
        super(ImageMaskDataset, self).__init__(root='data/', transform=transform)
        
        self.split_type = split_type
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
        
        # Assign labels based on the directory name
        self.labels = [self.class_to_idx[self._extract_class_name_from_path(path)] for path in self.image_paths]
        
        # Split data into train and validation sets
        self._split_data(train_percentage, val_percentage, seed)

        # Define a default transform if none is provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def _local_data_exists(self):
        # Check if local data exists by verifying if the 'data/' directory has any images
        if not os.path.exists(self.root):
            return False
        for class_name in os.listdir(self.root):
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path) and any(fname.endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(class_path)):
                return True
        return False
        
    def _download_and_extract_tar(self):
        # Define the local tar path under 'data_tar/'
        self.local_tar_path = os.path.join('data_tar', 'fish_data.tar.gz')
        self.root = 'data_tar/'  # Set the root for extracted data

        # Ensure the 'data_tar/' directory exists
        os.makedirs('data_tar', exist_ok=True)

        # Check if the tar file already exists locally
        if os.path.exists(self.local_tar_path):
            print(f"Tar file already exists locally at {self.local_tar_path}. Skipping download.")
        else:
            # Download the tar file from GCS
            print(f"Downloading {self.tar_prefix} from GCS to {self.local_tar_path}...")
            blob = self.bucket.blob(self.tar_prefix)
            blob.download_to_filename(self.local_tar_path)
            print(f"Downloaded {self.tar_prefix} to {self.local_tar_path}")

        # Extract the tar file to the 'data_tar/' directory
        print(f"Extracting {self.local_tar_path}...")
        with tarfile.open(self.local_tar_path, 'r:gz') as tar:
            tar.extractall(path=self.root)

        # Debug: Print the complete directory structure after extraction
        print("Complete directory structure after extraction:")
        # for root, dirs, files in os.walk(self.root):
        #     print(f"Directory: {root}")
        #     print(f"Subdirectories: {dirs}")
        #     print(f"Files: {files}")

        # Handle nested directories (e.g., 'data_tar/Sea Bass/Sea Bass/')
        for class_name in os.listdir(self.root):
            class_dir = os.path.join(self.root, class_name)
            if os.path.isdir(class_dir) and "GT" not in class_name:
                nested_dirs = os.listdir(class_dir)
                if len(nested_dirs) == 1 and os.path.isdir(os.path.join(class_dir, nested_dirs[0])):
                    nested_dir = os.path.join(class_dir, nested_dirs[0])
                    print(f"Detected nested directory: {nested_dir}. Moving contents to {class_dir}")
                    # Move all files up one level
                    for item in os.listdir(nested_dir):
                        shutil.move(os.path.join(nested_dir, item), class_dir)
                    # Remove the empty nested directory
                    shutil.rmtree(nested_dir)

        # Skip removing the tar file as requested
        print(f"Tar file retained at {self.local_tar_path}")


    def _get_local_image_paths(self):
        # Recursively collect all image file paths from 'data_tar/' directory, excluding "GT" directories
        local_image_paths = []
        print("Collecting image paths from the 'data_tar/' directory...")
        for root, dirs, files in os.walk(self.root):
            # Skip "GT" directories
            if "GT" in root:
                continue

            # Only scan for images in directories that are not "GT"
            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file_name)
                    local_image_paths.append(image_path)
                    #print(f"Found image: {image_path}")

        print(f"Found {len(local_image_paths)} images excluding GT directories.")
        return local_image_paths


    
    def _extract_class_name_from_path(self, path):
        # Extracts class name from directory structure, assuming format "data/class_name/image.jpg"
        return os.path.basename(os.path.dirname(path))

    def _split_data(self, train_percentage, val_percentage, seed):
        total_size = len(self.image_paths)
        if total_size == 0:
            raise ValueError("Dataset has no images. Check your local 'data/' directory.")

        indices = list(range(total_size))
        random.seed(seed)
        random.shuffle(indices)
        
        train_split_idx = int(total_size * train_percentage)
        val_split_idx = train_split_idx + int(total_size * val_percentage)
        
        if self.split_type == 'train':
            self.indices = indices[:train_split_idx]
        elif self.split_type == 'val':
            self.indices = indices[train_split_idx:val_split_idx]
        elif self.split_type == 'test':
            self.indices = indices[val_split_idx:]
        else:
            raise ValueError("Invalid split_type. Choose from 'train', 'val', or 'test'.")
        
        print(f"{self.split_type.capitalize()} split has {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map idx to the correct subset index
        real_idx = self.indices[idx]
        
        # Retrieve image path and label
        img_path = self.image_paths[real_idx]
        label = self.labels[real_idx]

        # Load image from local
        try:
            # Attempt to open the image
            image = Image.open(img_path).convert('RGB')
        except (SyntaxError, OSError) as e:
            print(f"Warning: Skipping corrupted image file {img_path}. Error: {e}")
            # Skip this index by returning the next valid item
            return self.__getitem__((idx + 1) % len(self.indices))
        
        # Apply resizing transform
        if self.transform:
            image = self.transform(image)

        return image, label


# Example usage:
if __name__ == '__main__':
    train_dataset = ImageMaskDataset(bucket_name='fish-dataset-cl', split_type='train', train_percentage=0.7, val_percentage=0.15, seed=42, image_size=96, tar_prefix='data_tar/fish_data.tar.gz', download=False)
    val_dataset = ImageMaskDataset(bucket_name='fish-dataset-cl', split_type='val', train_percentage=0.7, val_percentage=0.15, seed=42, image_size=96, tar_prefix='data_tar/fish_data.tar.gz', download=False)
    test_dataset = ImageMaskDataset(bucket_name='fish-dataset-cl', split_type='test', train_percentage=0.7, val_percentage=0.15, seed=42, image_size=96, tar_prefix='data_tar/fish_data.tar.gz', download=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
    import ipdb;ipdb.set_trace()
