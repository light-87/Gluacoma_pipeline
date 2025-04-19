"""
Data Augmentation Module

Implements augmentation strategies for glaucoma image data using Albumentations.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Union, Callable

def get_training_augmentations(
    image_size: Tuple[int, int] = (224, 224),
    rotation_range: float = 15.0,
    shift_range: float = 0.1,
    scale_range: float = 0.1,
    brightness_contrast_range: float = 0.2,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    normalization: str = 'imagenet'
) -> A.Compose:
    """
    Get augmentation pipeline for training data.
    
    Args:
        image_size: Target image size (width, height)
        rotation_range: Maximum rotation angle in degrees
        shift_range: Maximum shift as a fraction of image size
        scale_range: Maximum scale change as a fraction
        brightness_contrast_range: Maximum brightness/contrast change
        horizontal_flip: Whether to include horizontal flips
        vertical_flip: Whether to include vertical flips
        normalization: Normalization type ('imagenet', 'simple', or None)
        
    Returns:
        Albumentations Compose object with augmentations
    """
    # Define normalization parameters
    if normalization == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalization == 'simple':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = std = None
    
    # Create list of augmentations
    transforms = []
    
    # Spatial transforms
    if rotation_range > 0:
        transforms.append(A.Rotate(limit=rotation_range, p=0.7))
    
    if shift_range > 0:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=shift_range,
            scale_limit=scale_range,
            rotate_limit=0,  # We already have rotation
            p=0.5
        ))
    
    # Flips
    if horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))
    
    # Color transforms (only applied to image, not mask)
    transforms.append(A.RandomBrightnessContrast(
        brightness_limit=brightness_contrast_range,
        contrast_limit=brightness_contrast_range,
        p=0.5
    ))
    
    transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
    
    # Add normalization
    if mean is not None and std is not None:
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_validation_augmentations(
    image_size: Tuple[int, int] = (224, 224),
    normalization: str = 'imagenet'
) -> A.Compose:
    """
    Get augmentation pipeline for validation/test data.
    
    Args:
        image_size: Target image size (width, height)
        normalization: Normalization type ('imagenet', 'simple', or None)
        
    Returns:
        Albumentations Compose object with minimal augmentations
    """
    # Define normalization parameters
    if normalization == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalization == 'simple':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = std = None
    
    # Create list of validation transforms (minimal)
    transforms = []
    
    # Add normalization
    if mean is not None and std is not None:
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_augmentations(
    config,
    is_train: bool = True
) -> A.Compose:
    """
    Get augmentations based on configuration.
    
    Args:
        config: Configuration object with augmentation settings
        is_train: Whether to get training or validation augmentations
        
    Returns:
        Albumentations Compose object with augmentations
    """
    if is_train and config.augmentation.enabled:
        return get_training_augmentations(
            image_size=tuple(config.data.image_size),
            rotation_range=config.augmentation.rotation_range,
            shift_range=config.augmentation.width_shift_range,
            scale_range=config.augmentation.zoom_range,
            brightness_contrast_range=0.2,  # Fixed value
            horizontal_flip=config.augmentation.horizontal_flip,
            vertical_flip=config.augmentation.vertical_flip,
            normalization='imagenet'  # Fixed value
        )
    else:
        return get_validation_augmentations(
            image_size=tuple(config.data.image_size),
            normalization='imagenet'  # Fixed value
        )


if __name__ == "__main__":
    # Test code
    print("Available augmentations:")
    
    # Training augmentations
    train_aug = get_training_augmentations()
    print(f"Training augmentations: {train_aug}")
    
    # Validation augmentations
    val_aug = get_validation_augmentations()
    print(f"Validation augmentations: {val_aug}")
    
    # Example of applying augmentations
    import cv2
    import matplotlib.pyplot as plt
    
    # Create a dummy image and mask
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:150, 50:150] = 1  # Create a square mask
    
    # Apply augmentations
    augmented = train_aug(image=image, mask=mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    print(f"Original image shape: {image.shape}")
    print(f"Augmented image shape: {aug_image.shape}")
    print(f"Original mask shape: {mask.shape}")
    print(f"Augmented mask shape: {aug_mask.shape}")