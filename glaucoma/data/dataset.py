"""
Dataset Module

Implements memory-efficient PyTorch datasets for glaucoma image data.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import OrderedDict
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset")


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.
    
    This cache maintains a fixed size by removing the least recently used items
    when the capacity is reached.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the LRU cache.
        
        Args:
            capacity: Maximum number of items to store in the cache
        """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get an item from the cache.
        
        Args:
            key: Key to retrieve
            
        Returns:
            The cached item or None if not found
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        
    def put(self, key: str, value: np.ndarray) -> None:
        """
        Add an item to the cache.
        
        Args:
            key: Key for the item
            value: Item to cache
        """
        with self.lock:
            # If capacity is 0, don't store anything
            if self.capacity <= 0:
                return
                
            if key in self.cache:
                # Remove existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used item (first item)
                # Only try to pop if cache is not empty
                if self.cache:
                    self.cache.popitem(last=False)
                    
            # Add new item
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
            
    def __len__(self) -> int:
        """Get the number of items in the cache."""
        return len(self.cache)


class SquareImageDataset(Dataset):
    """
    Memory-efficient dataset for square fundus images.
    
    Features:
    - Lazy loading of images
    - LRU caching for frequently accessed images
    - Optional prefetching for improved performance
    - Support for optional transformations
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        cache_size: int = 100,
        prefetch_size: int = 0,
        mode: str = 'segmentation'
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with image and mask paths
            transform: Callable transformation to apply to images and masks
            target_size: Target size for resizing (width, height)
            cache_size: Maximum number of images to cache in memory
            prefetch_size: Number of images to prefetch (0 to disable)
            mode: 'segmentation' or 'classification'
        """
        self.data = data
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        
        # Initialize caches for images and masks
        self.image_cache = LRUCache(cache_size)
        self.mask_cache = LRUCache(cache_size) if mode == 'segmentation' else None
        
        # Set up prefetching if enabled
        self.prefetch_size = prefetch_size
        self.prefetch_queue = None
        self.prefetch_thread = None
        self.stop_prefetching = threading.Event()
        
        if prefetch_size > 0:
            self._setup_prefetching()
    
    def _setup_prefetching(self):
        """Set up background prefetching thread."""
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_size)
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self.prefetch_thread.start()
        logger.info(f"Started prefetching with queue size {self.prefetch_size}")
    
    def _prefetch_worker(self):
        """Background worker for prefetching images."""
        try:
            indices = np.random.permutation(len(self.data))
            while not self.stop_prefetching.is_set():
                for idx in indices:
                    if self.stop_prefetching.is_set():
                        break
                        
                    # Skip if already in cache
                    img_path = self.data.iloc[idx]['image_path']
                    if self.image_cache.get(img_path) is not None:
                        continue
                    
                    try:
                        # Load image
                        image = self._load_image(img_path)
                        
                        # Add to cache
                        self.image_cache.put(img_path, image)
                        
                        # If in segmentation mode, also load mask
                        if self.mode == 'segmentation':
                            mask_path = self.data.iloc[idx]['mask_path']
                            if self.mask_cache.get(mask_path) is None:
                                mask = self._load_mask(mask_path)
                                self.mask_cache.put(mask_path, mask)
                        
                        # Signal successful prefetch
                        self.prefetch_queue.put(idx, block=False)
                    except queue.Full:
                        # Queue is full, continue to next item
                        pass
                    except Exception as e:
                        logger.warning(f"Error prefetching item {idx}: {e}")
                
                # Shuffle indices for next round
                indices = np.random.permutation(len(self.data))
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array
        """
        try:
            # Use OpenCV for efficient loading
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image with OpenCV: {image_path}")
                # Fallback to PIL
                image = np.array(Image.open(image_path).convert('RGB'))
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image in case of error
            return np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load a mask from disk.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Mask as numpy array
        """
        try:
            # Use OpenCV for efficient loading (grayscale)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask with OpenCV: {mask_path}")
                # Fallback to PIL
                mask = np.array(Image.open(mask_path).convert('L'))
            
            # Resize to target size
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize to 0-1
            if mask.max() > 1:
                mask = mask / 255.0
            
            # Ensure binary values
            mask = (mask > 0.5).astype(np.float32)
            
            # Add channel dimension
            mask = np.expand_dims(mask, axis=0)
            
            return mask
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            # Return an empty mask in case of error
            return np.zeros((1, self.target_size[1], self.target_size[0]), dtype=np.float32)
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask) tensors
        """
        # Get paths
        row = self.data.iloc[idx]
        image_path = row['image_path']
        
        # Try to get image from cache
        image = self.image_cache.get(image_path)
        if image is None:
            # Load and cache image
            image = self._load_image(image_path)
            self.image_cache.put(image_path, image)
        
        # For segmentation mode, get mask
        if self.mode == 'segmentation':
            mask_path = row['mask_path']
            
            # Try to get mask from cache
            mask = self.mask_cache.get(mask_path)
            if mask is None:
                # Load and cache mask
                mask = self._load_mask(mask_path)
                self.mask_cache.put(mask_path, mask)
        else:
            # For classification, use label
            mask = np.array([row.get('label', 0)], dtype=np.float32)
        
        # Apply transformations if provided
        if self.transform:
            if self.mode == 'segmentation':
                # For segmentation, transform both image and mask
                transformed = self.transform(image=image, mask=mask[0])
                image = transformed["image"]
                mask = transformed["mask"]
                
                # Ensure mask is tensor with channel dimension
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).unsqueeze(0)
                elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
                    mask = mask.unsqueeze(0)
            else:
                # For classification, transform only image
                transformed = self.transform(image=image)
                image = transformed["image"]
                
                # Ensure label is tensor
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
        else:
            # Convert to tensors
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            if isinstance(mask, np.ndarray):
                if self.mode == 'segmentation':
                    mask = torch.from_numpy(mask)
                else:
                    mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask
    
    def cleanup(self):
        """Clean up resources."""
        # Stop prefetching
        if self.prefetch_thread is not None:
            self.stop_prefetching.set()
            self.prefetch_thread.join(timeout=1.0)
            logger.info("Stopped prefetching thread")
        
        # Clear caches
        self.image_cache.clear()
        if self.mask_cache:
            self.mask_cache.clear()
        logger.info("Cleared dataset caches")


def create_dataloaders(
    df: pd.DataFrame,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224),
    cache_size: int = 100,
    prefetch_size: int = 50,
    mode: str = 'segmentation'
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        df: DataFrame with image data and 'split' column
        transform_train: Transformations for training data
        transform_val: Transformations for validation/test data
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Image target size (width, height)
        cache_size: Cache size for each dataset
        prefetch_size: Number of images to prefetch
        mode: 'segmentation' or 'classification'
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Ensure 'split' column exists
    if 'split' not in df.columns:
        logger.warning("No 'split' column found in DataFrame")
        from sklearn.model_selection import train_test_split
        
        # Create splits (70/15/15)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Assign splits
        df = df.copy()
        df.loc[train_df.index, 'split'] = 'train'
        df.loc[val_df.index, 'split'] = 'val'
        df.loc[test_df.index, 'split'] = 'test'
    else:
        # Filter out rows with missing split
        if df['split'].isna().any():
            logger.warning(f"Found {df['split'].isna().sum()} rows with missing split")
            df = df.dropna(subset=['split'])
    
    # Get dataframes for each split
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    logger.info(f"Creating datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets
    train_dataset = SquareImageDataset(
        train_df,
        transform=transform_train,
        target_size=target_size,
        cache_size=cache_size,
        prefetch_size=prefetch_size,
        mode=mode
    )
    
    val_dataset = SquareImageDataset(
        val_df,
        transform=transform_val,
        target_size=target_size,
        cache_size=cache_size,
        prefetch_size=prefetch_size if len(val_df) > 0 else 0,
        mode=mode
    )
    
    test_dataset = SquareImageDataset(
        test_df,
        transform=transform_val,  # Same transform as validation
        target_size=target_size,
        cache_size=cache_size,
        prefetch_size=prefetch_size if len(test_df) > 0 else 0,
        mode=mode
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get statistics of a dataset.
    
    Args:
        df: DataFrame with dataset information
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_samples': len(df),
    }
    
    # Dataset distribution
    if 'dataset' in df.columns:
        stats['dataset_distribution'] = df['dataset'].value_counts().to_dict()
        
    # Split distribution
    if 'split' in df.columns:
        stats['split_distribution'] = df['split'].value_counts().to_dict()
        
        # Dataset distribution within splits
        if 'dataset' in df.columns:
            stats['dataset_per_split'] = {}
            for split in df['split'].unique():
                split_df = df[df['split'] == split]
                stats['dataset_per_split'][split] = (
                    split_df['dataset'].value_counts().to_dict()
                )
    
    # Image statistics
    if all(col in df.columns for col in ['image_width', 'image_height']):
        size_df = df.dropna(subset=['image_width', 'image_height'])
        if not size_df.empty:
            stats['image_size'] = {
                'width': {
                    'mean': size_df['image_width'].mean(),
                    'min': size_df['image_width'].min(),
                    'max': size_df['image_width'].max(),
                    'std': size_df['image_width'].std()
                },
                'height': {
                    'mean': size_df['image_height'].mean(),
                    'min': size_df['image_height'].min(),
                    'max': size_df['image_height'].max(),
                    'std': size_df['image_height'].std()
                }
            }
            
            # Aspect ratio
            size_df['aspect_ratio'] = size_df['image_width'] / size_df['image_height']
            stats['aspect_ratio'] = {
                'mean': size_df['aspect_ratio'].mean(),
                'min': size_df['aspect_ratio'].min(),
                'max': size_df['aspect_ratio'].max(),
                'std': size_df['aspect_ratio'].std()
            }
    
    return stats


if __name__ == "__main__":
    """Example usage."""
    import sys
    
    # Test with a sample CSV file
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} samples from {csv_path}")
            
            # Get statistics
            stats = get_dataset_statistics(df)
            print("\nDataset Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Create simple test dataset
            test_dataset = SquareImageDataset(
                df.head(10),  # Just use first 10 samples
                target_size=(224, 224),
                cache_size=5,
                prefetch_size=0,
                mode='segmentation'
            )
            
            # Test getting a sample
            image, mask = test_dataset[0]
            print(f"\nSample image shape: {image.shape}")
            print(f"Sample mask shape: {mask.shape}")
            
            # Create dataloaders
            dataloaders = create_dataloaders(
                df,
                batch_size=4,
                num_workers=0,  # Use 0 for debugging
                target_size=(224, 224),
                cache_size=10
            )
            
            print("\nDataLoader sizes:")
            for split, loader in dataloaders.items():
                print(f"  {split}: {len(loader)} batches")
        else:
            print(f"CSV file not found: {csv_path}")
    else:
        print("Please provide a CSV file path as argument")