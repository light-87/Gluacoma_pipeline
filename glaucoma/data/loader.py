"""
Data Loading Module

Handles loading and consolidating glaucoma datasets from different sources,
with a focus on memory efficiency and performance.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from typing import Dict, List, Optional, Union, Tuple
import logging
from tqdm import tqdm
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loader")


class DatasetLoader:
    """
    Handles loading of glaucoma datasets from standardized directory structure.
    
    This loader specifically focuses on square images for improved consistency.
    """
    
    def __init__(self, data_dirs: Dict[str, Dict[str, str]]):
        """
        Initialize the dataset loader.
        
        Args:
            data_dirs: Dictionary mapping dataset names to image and mask directories
                       Example: {'REFUGE': {'images': 'path/to/images', 'masks': 'path/to/masks'}}
        """
        self.data_dirs = data_dirs
        self.verify_dirs()
        
    def verify_dirs(self) -> None:
        """
        Verify that all specified directories exist.
        
        Logs warnings for missing directories.
        """
        for dataset, paths in self.data_dirs.items():
            for key, path in paths.items():
                if not os.path.exists(path):
                    logger.warning(f"Directory not found for {dataset} {key}: {path}")
                    
    def get_dataset_paths(self, dataset: str) -> Tuple[List[str], List[str]]:
        """
        Get lists of image and mask paths for a specific dataset.
        
        Args:
            dataset: Name of the dataset (e.g., 'REFUGE', 'ORIGA', 'G1020')
            
        Returns:
            Tuple of (image_paths, mask_paths)
        """
        if dataset not in self.data_dirs:
            logger.error(f"Dataset {dataset} not found in configuration")
            return [], []
            
        img_dir = self.data_dirs[dataset]['images']
        mask_dir = self.data_dirs[dataset]['masks']
        
        # Get all image paths
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        image_paths.extend(sorted(glob.glob(os.path.join(img_dir, "*.png"))))
        
        # Get corresponding mask paths
        mask_paths = []
        for img_path in image_paths:
            # Extract base filename without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Look for corresponding mask
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            
            # If not found, try other extensions
            if not os.path.exists(mask_path):
                alt_mask_path = os.path.join(mask_dir, f"{base_name}.jpg")
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                else:
                    mask_path = ""  # Empty string for missing masks
            
            mask_paths.append(mask_path)
        
        return image_paths, mask_paths
    
    def load_dataset(self, dataset: str) -> pd.DataFrame:
        """
        Load a single dataset.
        
        Args:
            dataset: Name of the dataset to load
            
        Returns:
            DataFrame with image and mask paths
        """
        logger.info(f"Loading {dataset} dataset")
        
        # Get paths
        image_paths, mask_paths = self.get_dataset_paths(dataset)
        
        if not image_paths:
            logger.warning(f"No images found for {dataset}")
            return pd.DataFrame()
            
        # Create list of dictionaries for each entry
        data = []
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            # Skip entries with missing masks
            if not mask_path or not os.path.exists(mask_path):
                logger.debug(f"Skipping {img_path} due to missing mask")
                continue
                
            # Check if both files exist
            if not os.path.exists(img_path):
                logger.debug(f"Image doesn't exist: {img_path}")
                continue
                
            # Create entry
            entry = {
                'filename': os.path.splitext(os.path.basename(img_path))[0],
                'image_path': img_path,
                'mask_path': mask_path,
                'dataset': dataset
            }
            data.append(entry)
                
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add basic image information
        if not df.empty:
            logger.info(f"Adding image information for {dataset}")
            # Add basic information for a sample of images to save time
            sample_size = min(100, len(df))
            sample_indices = np.random.choice(len(df), sample_size, replace=False)
            
            for idx in tqdm(sample_indices, desc=f"Sampling {dataset} images"):
                img_path = df.iloc[idx]['image_path']
                try:
                    # Use OpenCV for efficient image reading
                    img = cv2.imread(img_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        df.loc[df.index[idx], 'image_height'] = height
                        df.loc[df.index[idx], 'image_width'] = width
                except Exception as e:
                    logger.warning(f"Error reading image {img_path}: {e}")
        
        logger.info(f"Loaded {len(df)} valid samples from {dataset}")
        return df
    
    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load all datasets specified in the configuration.
        
        Returns:
            Consolidated DataFrame with all datasets
        """
        all_data = []
        
        for dataset in self.data_dirs:
            df = self.load_dataset(dataset)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            logger.warning("No valid data loaded from any dataset")
            return pd.DataFrame()
            
        # Concatenate all datasets
        consolidated_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Consolidated dataset contains {len(consolidated_df)} samples")
        
        # Log dataset distribution
        dataset_counts = consolidated_df['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            logger.info(f"  - {dataset}: {count} samples ({count/len(consolidated_df)*100:.1f}%)")
        
        return consolidated_df
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze a dataset for basic statistics.
        
        Args:
            df: DataFrame with image and mask paths
            
        Returns:
            Dictionary with dataset statistics
        """
        if df.empty:
            return {'status': 'empty'}
            
        stats = {
            'num_samples': len(df),
            'datasets': df['dataset'].value_counts().to_dict()
        }
        
        # Image dimensions
        if 'image_width' in df.columns and 'image_height' in df.columns:
            # Filter out rows with missing dimensions
            dim_df = df.dropna(subset=['image_width', 'image_height'])
            if not dim_df.empty:
                stats['avg_width'] = dim_df['image_width'].mean()
                stats['avg_height'] = dim_df['image_height'].mean()
                stats['min_width'] = dim_df['image_width'].min()
                stats['min_height'] = dim_df['image_height'].min()
                stats['max_width'] = dim_df['image_width'].max()
                stats['max_height'] = dim_df['image_height'].max()
        
        return stats


def save_dataset(df: pd.DataFrame, output_path: Union[str, Path], 
                create_splits: bool = False, train_ratio: float = 0.7, 
                val_ratio: float = 0.15, random_state: int = 42) -> None:
    """
    Save a dataset to CSV with optional train/val/test splits.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the CSV file
        create_splits: Whether to create train/val/test splits
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data (test_ratio = 1 - train_ratio - val_ratio)
        random_state: Random seed for reproducibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        logger.warning("Attempting to save an empty dataset")
        df.to_csv(output_path, index=False)
        return
    
    # Create splits if requested
    if create_splits:
        from sklearn.model_selection import train_test_split
        
        # Create stratification key based on dataset
        if 'dataset' in df.columns:
            df['stratify_key'] = df['dataset']
        else:
            df['stratify_key'] = 'default'
            
        # Split into train+val and test
        train_val_ratio = train_ratio + val_ratio
        train_val_df, test_df = train_test_split(
            df, test_size=(1 - train_val_ratio),
            random_state=random_state,
            stratify=df['stratify_key']
        )
        
        # Split train+val into train and val
        adjusted_val_ratio = val_ratio / train_val_ratio
        train_df, val_df = train_test_split(
            train_val_df, test_size=adjusted_val_ratio,
            random_state=random_state,
            stratify=train_val_df['stratify_key']
        )
        
        # Assign splits
        df.loc[train_df.index, 'split'] = 'train'
        df.loc[val_df.index, 'split'] = 'val'
        df.loc[test_df.index, 'split'] = 'test'
        
        # Remove temporary column
        df = df.drop(columns=['stratify_key'])
        
        # Log split distribution
        split_counts = df['split'].value_counts()
        logger.info("Dataset split distribution:")
        for split, count in split_counts.items():
            logger.info(f"  - {split}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved dataset with {len(df)} samples to {output_path}")
    
    
if __name__ == "__main__":
    # Example usage
    from glaucoma.config import Config
    
    # Create default config
    config = Config()
    
    # Create dataset loader
    loader = DatasetLoader(config.data.data_dirs)
    
    # Load all datasets
    df = loader.load_all_datasets()
    
    # Analyze dataset
    stats = loader.analyze_dataset(df)
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Save dataset with splits
    save_dataset(
        df, 
        'data/consolidated_dataset.csv',
        create_splits=True,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        random_state=config.data.random_state
    )