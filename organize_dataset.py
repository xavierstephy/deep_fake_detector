<<<<<<< HEAD
import os
import shutil
import random
from pathlib import Path

def organize_dataset(source_dir, output_dir, train_split=0.8, seed=42):
    """
    Organize dataset into train/val structure for ImageFolder.
    
    Args:
        source_dir: Directory containing numbered folders with images
        output_dir: Output directory where train/ and val/ folders will be created
        train_split: Fraction of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all numbered folders
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    folders.sort()
    
    print(f"Found {len(folders)} folders in {source_dir}")
    
    # Split folders into train/val
    random.shuffle(folders)
    split_idx = int(len(folders) * train_split)
    train_folders = folders[:split_idx]
    val_folders = folders[split_idx:]
    
    print(f"Training folders: {len(train_folders)}")
    print(f"Validation folders: {len(val_folders)}")
    
    # Copy folders to train/val directories
    for folder in train_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(train_dir, folder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"Copied {folder} to train/")
    
    for folder in val_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(val_dir, folder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"Copied {folder} to val/")
    
    print(f"\nDataset organized successfully!")
    print(f"Train: {len(train_folders)} folders")
    print(f"Val: {len(val_folders)} folders")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Configure paths
    source_dataset = "dataset"  # Your current dataset folder
    organized_dataset = "organized_dataset"  # New organized dataset
    
    organize_dataset(source_dataset, organized_dataset, train_split=0.8)
    
    print("\nNow you can train with:")
    print(f"python train_model.py --data-dir {organized_dataset} --device cpu --model resnet18 --epochs 10")
=======
import os
import shutil
import random
from pathlib import Path

def organize_dataset(source_dir, output_dir, train_split=0.8, seed=42):
    """
    Organize dataset into train/val structure for ImageFolder.
    
    Args:
        source_dir: Directory containing numbered folders with images
        output_dir: Output directory where train/ and val/ folders will be created
        train_split: Fraction of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all numbered folders
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    folders.sort()
    
    print(f"Found {len(folders)} folders in {source_dir}")
    
    # Split folders into train/val
    random.shuffle(folders)
    split_idx = int(len(folders) * train_split)
    train_folders = folders[:split_idx]
    val_folders = folders[split_idx:]
    
    print(f"Training folders: {len(train_folders)}")
    print(f"Validation folders: {len(val_folders)}")
    
    # Copy folders to train/val directories
    for folder in train_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(train_dir, folder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"Copied {folder} to train/")
    
    for folder in val_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(val_dir, folder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"Copied {folder} to val/")
    
    print(f"\nDataset organized successfully!")
    print(f"Train: {len(train_folders)} folders")
    print(f"Val: {len(val_folders)} folders")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Configure paths
    source_dataset = "dataset"  # Your current dataset folder
    organized_dataset = "organized_dataset"  # New organized dataset
    
    organize_dataset(source_dataset, organized_dataset, train_split=0.8)
    
    print("\nNow you can train with:")
    print(f"python train_model.py --data-dir {organized_dataset} --device cpu --model resnet18 --epochs 10")
>>>>>>> 0c57338d10119562221af77a603da511f81f8170
