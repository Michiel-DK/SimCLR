import os
import shutil
from PIL import Image
import hashlib

# Define source and target directories
source_dir = "data_tar"
target_dir = "data_no_dupli"

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

def get_file_hash(file_path):
    """Generate hash for a file to check for duplicates."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def remove_duplicates_and_copy(src_dir, tgt_dir):
    """Iterate through directories, remove duplicates, and save unique images."""
    file_hashes = {}
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Skip non-image files
            if not file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                continue

            full_path = os.path.join(root, file)
            file_hash = get_file_hash(full_path)

            if file_hash in file_hashes:
                print(f"Duplicate found: {full_path}, skipping...")
                continue
            
            # Save unique hash
            file_hashes[file_hash] = full_path

            # Determine relative path and corresponding mask
            rel_path = os.path.relpath(full_path, src_dir)
            new_path = os.path.join(tgt_dir, rel_path)

            # Ensure the target directory exists
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # Copy unique image
            shutil.copy2(full_path, new_path)

# Process both image and GT folders
remove_duplicates_and_copy(source_dir, target_dir)

print(f"All unique images and masks saved in {target_dir}.")
