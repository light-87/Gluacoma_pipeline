"""
Data Module

This module handles dataset creation, loading, and augmentation for the glaucoma detection pipeline.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
import threading
from queue import Queue
from functools import lru_cache
import time

class DatasetLoader:
    """Loads datasets from multiple directories."""
    
    def __init__(self, data_dirs: Dict[str, Dict[str, str]]):
        """Initialize the dataset loader.
        
        Args:
            data_dirs: Dictionary mapping dataset names to paths for images and masks
                       Example: {'REFUGE': {'images': 'path/to/images', 'masks': 'path/to/masks'}}
        """
        self.data_dirs = data_dirs
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            DataFrame with image and mask paths
        """
        if dataset_name not in self.data_dirs:
            raise ValueError(f"Dataset {dataset_name} not found in data_dirs")
        
        paths = self.data_dirs[dataset_name]
        if not os.path.exists(paths['images']) or not os.path.exists(paths['masks']):
            print(f"Warning: Paths for dataset {dataset_name} do not exist")
            return pd.DataFrame()
        
        img_dir = paths['images']
        mask_dir = paths['masks']
        
        # Get image and mask files
        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))
        
        # Collect paths with matching masks
        image_paths = []
        mask_paths = []
        dataset_labels = []
        
        for img_file in img_files:
            img_stem = os.path.splitext(img_file)[0]
            mask_file = next((m for m in mask_files if os.path.splitext(m)[0] == img_stem), None)
            
            if mask_file:
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                
                image_paths.append(img_path)
                mask_paths.append(mask_path)
                dataset_labels.append(dataset_name)
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'mask_path': mask_paths,
            'dataset': dataset_labels
        })
        
        print(f"Loaded {len(df)} samples from dataset {dataset_name}")
        return df
    
    def load_all_datasets(self) -> pd.DataFrame:
        """Load all datasets defined in data_dirs.
        
        Returns:
            DataFrame with combined datasets
        """
        datasets = []
        
        for dataset_name in self.data_dirs:
            df = self.load_dataset(dataset_name)
            if not df.empty:
                datasets.append(df)
        
        if not datasets:
            return pd.DataFrame()
        
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples from {len(datasets)} datasets")
        return combined_df
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Analyze the dataset to get statistics.
        
        Args:
            df: DataFrame with image and mask paths
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(df),
            'datasets': {}
        }
        
        if 'dataset' in df.columns:
            dataset_counts = df['dataset'].value_counts().to_dict()
            stats['datasets'] = dataset_counts
        
        if 'split' in df.columns:
            split_counts = df['split'].value_counts().to_dict()
            stats['splits'] = split_counts
        
        return stats

def create_dataset_splits(df: pd.DataFrame, 
                         validation_split: float = 0.15, 
                         test_split: float = 0.15, 
                         random_state: int = 42) -> pd.DataFrame:
    """Create train/validation/test splits for the dataset.
    
    Args:
        df: DataFrame with image and mask paths
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with added 'split' column
    """
    from sklearn.model_selection import train_test_split
    
    # If the split column already exists, return the DataFrame
    if 'split' in df.columns:
        return df
    
    # First split into train and temp (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=validation_split + test_split, 
        random_state=random_state, stratify=df.get('dataset', None)
    )
    
    # Then split temp into val and test
    val_size = validation_split / (validation_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_size, 
        random_state=random_state, stratify=temp_df.get('dataset', None)
    )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine back to full dataframe
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"Dataset splits: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    return full_df

def save_dataset(df: pd.DataFrame, 
                path: Union[str, Path], 
                create_splits: bool = True,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                random_state: int = 42):
    """Save dataset to a CSV file.
    
    Args:
        df: DataFrame with image and mask paths
        path: Path to save the dataset
        create_splits: Whether to create train/val/test splits
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation
        test_ratio: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Create splits if needed
    if create_splits and 'split' not in df.columns:
        # Adjust ratios to make sure they sum to 1
        total = train_ratio + val_ratio + test_ratio
        val_ratio = val_ratio / total
        test_ratio = test_ratio / total
        
        df = create_dataset_splits(
            df, validation_split=val_ratio, test_split=test_ratio, random_state=random_state
        )
    
    # Save dataset
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved dataset with {len(df)} samples to {path}")

def get_augmentations(config, is_train: bool = True):
    """Create data augmentations.
    
    Args:
        config: Preprocessing configuration
        is_train: Whether to create training augmentations or validation/test augmentations
        
    Returns:
        Albumentations transformations
    """
    # Get image size
    image_size = config.image_size
    
    # Set normalization values
    if config.normalization == "imagenet":
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
        std = [0.229, 0.224, 0.225]   # ImageNet std
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    if is_train and config.augmentation_enabled:
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_contrast_range,
                contrast_limit=config.brightness_contrast_range,
                p=0.2
            ),
            A.ShiftScaleRotate(
                shift_limit=config.shift_range, 
                scale_limit=config.scale_range, 
                rotate_limit=config.rotation_range, 
                p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    return transform

class PrefetchQueue:
    """Helper class for prefetching images in background threads."""
    
    def __init__(self, max_size: int = 10):
        """Initialize prefetch queue.
        
        Args:
            max_size: Maximum size of the queue
        """
        self.queue = Queue(maxsize=max_size)
        self.should_stop = threading.Event()
    
    def prefetch_worker(self, indices: List[int], dataset):
        """Worker function to prefetch data.
        
        Args:
            indices: List of indices to prefetch
            dataset: Dataset to fetch data from
        """
        for idx in indices:
            if self.should_stop.is_set():
                break
            try:
                data = dataset._load_data(idx)
                if not self.queue.full() and not self.should_stop.is_set():
                    self.queue.put((idx, data), block=False)
            except Exception as e:
                print(f"Error prefetching data for index {idx}: {e}")
    
    def stop(self):
        """Stop prefetching."""
        self.should_stop.set()

class GlaucomaDataset(Dataset):
    """Dataset for glaucoma segmentation with efficient data loading."""
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        transform: Optional[A.Compose] = None,
        cache_size: int = 100,
        prefetch_size: int = 0,
        target_size: Tuple[int, int] = (224, 224),
        mode: str = 'segmentation'
    ):
        """Initialize dataset.
        
        Args:
            dataframe: DataFrame with image and mask paths
            transform: Albumentations transformations
            cache_size: Size of the LRU cache (0 to disable)
            prefetch_size: Number of samples to prefetch (0 to disable)
            target_size: Target size for images and masks
            mode: Dataset mode ('segmentation' or 'classification')
        """
        self.dataframe = dataframe
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        
        # Setup caching
        self.cache_size = cache_size
        self.data_cache = {}  # Use a simple dictionary for caching instead of lru_cache
        
        # Setup prefetching
        self.prefetch_size = prefetch_size
        self.prefetch_queue = None
        if prefetch_size > 0:
            self.prefetch_queue = PrefetchQueue(max_size=prefetch_size)
            self.prefetch_thread = None
            self._start_prefetching()
    
    def _start_prefetching(self):
        """Start prefetching thread."""
        if self.prefetch_queue is None:
            return
        
        # Make sure we're not restarting an active thread
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return
        
        # Select first batch of indices to prefetch
        indices = list(range(min(self.prefetch_size, len(self.dataframe))))
        
        # Start prefetching thread
        self.prefetch_thread = threading.Thread(
            target=self.prefetch_queue.prefetch_worker,
            args=(indices, self)
        )
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
    
    def _load_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask for a given index.
        
        Args:
            idx: Index to load
            
        Returns:
            Tuple of (image, mask) as numpy arrays
        """
        # Check cache first if caching is enabled
        if self.cache_size > 0 and idx in self.data_cache:
            return self.data_cache[idx]
            
        # Get paths
        img_path = self.dataframe.iloc[idx]['image_path']
        mask_path = self.dataframe.iloc[idx]['mask_path']
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.float32)
        
        # Cache the result if caching is enabled
        if self.cache_size > 0:
            # Maintain cache size
            if len(self.data_cache) >= self.cache_size:
                # Remove a random key if we're at capacity
                oldest_key = next(iter(self.data_cache))
                self.data_cache.pop(oldest_key)
            self.data_cache[idx] = (image, mask)
        
        return image, mask
    
    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index to get
            
        Returns:
            Tuple of (image, mask) as tensors
        """
        # Get image and mask
        if self.prefetch_queue is not None:
            try:
                # Try to get from prefetch queue first
                prefetched_data = None
                
                # Check if this index is in the queue
                for _ in range(self.prefetch_queue.queue.qsize()):
                    queue_idx, data = self.prefetch_queue.queue.get(block=False)
                    if queue_idx == idx:
                        prefetched_data = data
                    else:
                        # Put it back in the queue
                        self.prefetch_queue.queue.put((queue_idx, data))
                
                if prefetched_data:
                    # Schedule next batch of indices to prefetch
                    next_indices = [
                        (idx + i) % len(self) 
                        for i in range(1, self.prefetch_size + 1)
                    ]
                    threading.Thread(
                        target=self.prefetch_queue.prefetch_worker,
                        args=(next_indices, self),
                        daemon=True
                    ).start()
                    
                    # Use the prefetched data
                    image, mask = prefetched_data
                else:
                    # If not in queue, load directly
                    image, mask = self._load_data(idx)
            except Exception as e:
                # Fall back to direct loading if queue fails
                image, mask = self._load_data(idx)
        else:
            # Direct loading if prefetching is disabled
            image, mask = self._load_data(idx)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension to mask if needed
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        if self.mode == 'segmentation' and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask
    
    def cleanup(self):
        """Clean up resources."""
        if self.prefetch_queue is not None:
            self.prefetch_queue.stop()
            if self.prefetch_thread is not None:
                self.prefetch_thread.join(timeout=0.5)
    
    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries
        state['prefetch_queue'] = None
        state['prefetch_thread'] = None
        state['data_cache'] = {}  # Don't pickle the cache
        return state
    
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        # Initialize new cache and prefetch queue if needed
        self.data_cache = {}
        if self.prefetch_size > 0:
            self.prefetch_queue = PrefetchQueue(max_size=self.prefetch_size)
            self._start_prefetching()

def create_dataloaders(
    df: pd.DataFrame,
    transform_train: A.Compose,
    transform_val: A.Compose,
    batch_size: int = 16,
    num_workers: int = 0,  # Default to 0 workers to avoid multiprocessing issues
    target_size: Tuple[int, int] = (224, 224),
    cache_size: int = 100,
    prefetch_size: int = 0,
    mode: str = 'segmentation'
) -> Dict[str, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        df: DataFrame with image and mask paths and split column
        transform_train: Transformations for training data
        transform_val: Transformations for validation and test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        target_size: Target size for images and masks
        cache_size: Size of the LRU cache
        prefetch_size: Number of samples to prefetch
        mode: Dataset mode ('segmentation' or 'classification')
        
    Returns:
        Dictionary with 'train', 'val', and 'test' dataloaders
    """
    # Make sure df has a split column
    if 'split' not in df.columns:
        df = create_dataset_splits(df)
    
    # Create datasets
    train_dataset = GlaucomaDataset(
        df[df['split'] == 'train'],
        transform=transform_train,
        cache_size=cache_size,
        prefetch_size=prefetch_size,
        target_size=target_size,
        mode=mode
    )
    
    val_dataset = GlaucomaDataset(
        df[df['split'] == 'val'],
        transform=transform_val,
        cache_size=cache_size,
        prefetch_size=prefetch_size,
        target_size=target_size,
        mode=mode
    )
    
    test_dataset = GlaucomaDataset(
        df[df['split'] == 'test'],
        transform=transform_val,
        cache_size=cache_size,
        prefetch_size=prefetch_size,
        target_size=target_size,
        mode=mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }