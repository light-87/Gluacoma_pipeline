# Model Results Analysis

This notebook provides comprehensive analysis of glaucoma detection model performance. It includes visualization of results, detailed metrics examination, and error analysis to help understand model strengths and weaknesses.

## Setup and Configuration

```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add project root to path
sys.path.append("..")

# Import project modules
from glaucoma.evaluation.metrics import (
    calculate_metrics, calculate_confusion_matrix_elements,
    calculate_roc_curve, calculate_pr_curve, calculate_cdr
)
from glaucoma.evaluation.visualization import VisualizationManager
from glaucoma.models.factory import create_model
from glaucoma.evaluation.evaluator import Evaluator
from glaucoma.utils.logging import PipelineLogger
from glaucoma.config import load_config

# Configure matplotlib
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# Set the paths and configuration
config_path = "../config.yaml"  # Path to your configuration file
results_dir = "../results"  # Path to results directory
models_dir = "../checkpoints"  # Path to saved models

# Load configuration
config = load_config(config_path)

# Setup logger
logger = PipelineLogger('model_analysis')
```

## Load Model and Results

```python
def load_saved_results(results_path: str) -> Dict[str, Any]:
    """Load saved evaluation results."""
    with open(results_path, 'r') as f:
        return json.load(f)

def load_model_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a saved model checkpoint."""
    return torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Load results
results_path = os.path.join(results_dir, "evaluation_results.json")
results = load_saved_results(results_path)

# Load model if needed for additional analysis
checkpoint_path = os.path.join(models_dir, "best_model.pt")
if os.path.exists(checkpoint_path):
    checkpoint = load_model_checkpoint(checkpoint_path)
    model_config = checkpoint['config']
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model = None
    print("Model checkpoint not found. Analysis will be limited to saved results.")

# Print basic results
print(f"Loaded results: {results.keys()}")
print(f"Dice Score: {results.get('dice', 'N/A'):.4f}")
print(f"IoU Score: {results.get('iou', 'N/A'):.4f}")
print(f"Precision: {results.get('precision', 'N/A'):.4f}")
print(f"Recall: {results.get('recall', 'N/A'):.4f}")
```

## Analyze Metrics

```python
# Create a DataFrame of metrics for easier analysis
metrics_df = pd.DataFrame({
    'Metric': ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1'],
    'Value': [
        results.get('dice', 0), 
        results.get('iou', 0), 
        results.get('accuracy', 0), 
        results.get('precision', 0), 
        results.get('recall', 0), 
        results.get('f1', 0)
    ]
})

# Plot metrics
plt.figure(figsize=(10, 6))
bar = sns.barplot(x='Metric', y='Value', data=metrics_df)
plt.title('Model Performance Metrics')
plt.ylim(0, 1)

# Add value labels to the bars
for p in bar.patches:
    bar.annotate(f'{p.get_height():.4f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

plt.tight_layout()
plt.show()

# Extract and plot the confusion matrix
conf_matrix = np.array(results.get('confusion_matrix', [[0, 0], [0, 0]]))
tn, fp, fn, tp = conf_matrix.flatten()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Calculate some derived metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
balanced_accuracy = (results.get('recall', 0) + specificity) / 2

print(f"Specificity: {specificity:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
```

## Training History Analysis

```python
# If you have training history available, load and plot it
try:
    if checkpoint and 'metrics_history' in checkpoint:
        history = checkpoint['metrics_history']
        epochs = list(range(1, len(history['train_loss']) + 1))

        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation Dice score
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_dice'], label='Training Dice')
        plt.plot(epochs, history['val_dice'], label='Validation Dice')
        plt.title('Training and Validation Dice Score')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot learning rate
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.tight_layout()
        plt.show()
    else:
        print("No training history found in checkpoint.")
except Exception as e:
    print(f"Error loading training history: {e}")
```

## ROC and PR Curves

```python
# If you have ROC or PR curve data available
try:
    # Load or compute curves
    viz_manager = VisualizationManager(results_dir)
    
    # Load images for ROC and PR curves
    roc_curve_path = os.path.join(results_dir, "roc_curve.png")
    pr_curve_path = os.path.join(results_dir, "pr_curve.png")
    
    # Display images if they exist
    if os.path.exists(roc_curve_path):
        roc_img = Image.open(roc_curve_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(np.array(roc_img))
        plt.axis('off')
        plt.title('ROC Curve')
        plt.show()
        
    if os.path.exists(pr_curve_path):
        pr_img = Image.open(pr_curve_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(np.array(pr_img))
        plt.axis('off')
        plt.title('Precision-Recall Curve')
        plt.show()
except Exception as e:
    print(f"Error loading curve visualizations: {e}")
```

## Sample Predictions Analysis

```python
# Load sample prediction images
sample_preds_path = os.path.join(results_dir, "sample_predictions.png")

if os.path.exists(sample_preds_path):
    preds_img = Image.open(sample_preds_path)
    plt.figure(figsize=(15, 10))
    plt.imshow(np.array(preds_img))
    plt.axis('off')
    plt.title('Sample Predictions')
    plt.show()
else:
    print("Sample predictions image not found.")
```

## Model Performance by Image Characteristics

```python
# This section requires test dataset and model for inference
# It analyzes model performance based on image characteristics
# such as contrast, brightness, presence of artifacts, etc.

# Example code (requires actual test data)
"""
def analyze_performance_by_characteristic(model, test_loader, characteristic_fn):
    '''Analyze model performance grouped by image characteristic.'''
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    results = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Move to CPU for analysis
            images_cpu = images.cpu()
            masks_cpu = masks.cpu()
            preds_cpu = preds.cpu()
            
            # Calculate characteristic and performance
            for i in range(images.size(0)):
                characteristic = characteristic_fn(images_cpu[i])
                metrics = calculate_metrics(preds_cpu[i], masks_cpu[i])
                
                results.append({
                    'characteristic': characteristic,
                    'dice': metrics['dice'],
                    'iou': metrics['iou']
                })
    
    return pd.DataFrame(results)

# Example characteristic functions
def calculate_contrast(image):
    '''Calculate image contrast.'''
    img = image[0].numpy()  # Assuming grayscale, take first channel
    return np.std(img)

def calculate_brightness(image):
    '''Calculate image brightness.'''
    img = image[0].numpy()
    return np.mean(img)

# Run analysis if model is available
if model is not None:
    # Create a test loader with actual data
    # test_loader = ...
    
    # Analyze by contrast
    contrast_df = analyze_performance_by_characteristic(model, test_loader, calculate_contrast)
    contrast_df['contrast_bin'] = pd.cut(contrast_df['contrast'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='contrast_bin', y='dice', data=contrast_df)
    plt.title('Dice Score by Image Contrast')
    plt.xlabel('Contrast Level')
    plt.ylabel('Dice Score')
    plt.tight_layout()
    plt.show()
"""
```

## Error Analysis

```python
# Function to identify challenging cases - images where model performs poorly
# This requires sample images and predictions from the evaluation

def find_challenging_cases(model, test_loader, threshold=0.5):
    """Find images where model performs poorly."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    challenging_cases = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Calculate dice score
            for i in range(images.size(0)):
                metrics = calculate_metrics(preds[i].cpu(), masks[i].cpu())
                
                # If performance is poor, save for analysis
                if metrics['dice'] < threshold:
                    challenging_cases.append({
                        'image': images[i].cpu(),
                        'mask': masks[i].cpu(),
                        'pred': preds[i].cpu(),
                        'metrics': metrics
                    })
    
    return challenging_cases

# Example of analysis if model and data are available
"""
if model is not None:
    # Create test loader with actual data
    # test_loader = ...
    
    # Find challenging cases
    challenging_cases = find_challenging_cases(model, test_loader, threshold=0.3)
    
    # Display a few challenging cases
    if challenging_cases:
        num_to_show = min(5, len(challenging_cases))
        plt.figure(figsize=(15, 4 * num_to_show))
        
        for i in range(num_to_show):
            case = challenging_cases[i]
            
            # Original image
            plt.subplot(num_to_show, 3, i*3 + 1)
            plt.imshow(case['image'][0].numpy(), cmap='gray')  # Assuming grayscale
            plt.title(f"Image (Dice: {case['metrics']['dice']:.3f})")
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(num_to_show, 3, i*3 + 2)
            plt.imshow(case['mask'][0].numpy(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Prediction
            plt.subplot(num_to_show, 3, i*3 + 3)
            plt.imshow(case['pred'][0].numpy() > 0.5, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
"""
```

## Statistical Analysis of Results

```python
# Statistical analysis of image-wise performance
# This requires results for individual test images

def analyze_result_distribution(metrics_list):
    """Analyze the distribution of metrics across test images."""
    metrics_df = pd.DataFrame(metrics_list)
    
    # Summary statistics
    summary = metrics_df.describe()
    print("Summary Statistics:")
    print(summary)
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(metrics_df['dice'], kde=True)
    plt.title('Distribution of Dice Scores')
    
    plt.subplot(2, 2, 2)
    sns.histplot(metrics_df['iou'], kde=True)
    plt.title('Distribution of IoU Scores')
    
    plt.subplot(2, 2, 3)
    sns.histplot(metrics_df['precision'], kde=True)
    plt.title('Distribution of Precision')
    
    plt.subplot(2, 2, 4)
    sns.histplot(metrics_df['recall'], kde=True)
    plt.title('Distribution of Recall')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation between metrics
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.show()
    
    return summary

# Example of usage if you have per-image metrics
"""
# Assume metrics_per_image is a list of dictionaries with metrics for each test image
metrics_per_image = []
summary = analyze_result_distribution(metrics_per_image)
"""
```

## Model Performance at Different Thresholds

```python
# Explore model performance at different threshold values
def analyze_threshold_sensitivity(preds, targets, thresholds=None):
    """Analyze how model performance varies with different thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    # Performance metrics at different thresholds
    results = []
    for threshold in thresholds:
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        
        for pred, target in zip(preds, targets):
            metrics = calculate_metrics(pred, target, threshold=threshold)
            dice_scores.append(metrics['dice'])
            iou_scores.append(metrics['iou'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
        
        results.append({
            'threshold': threshold,
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['threshold'], df['dice'])
    plt.title('Dice Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    
    plt.subplot(2, 2, 2)
    plt.plot(df['threshold'], df['iou'])
    plt.title('IoU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    
    plt.subplot(2, 2, 3)
    plt.plot(df['threshold'], df['precision'])
    plt.title('Precision vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    
    plt.subplot(2, 2, 4)
    plt.plot(df['threshold'], df['recall'])
    plt.title('Recall vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal threshold for Dice score
    best_dice_idx = df['dice'].idxmax()
    best_dice = df.iloc[best_dice_idx]
    print(f"Optimal threshold for Dice score: {best_dice['threshold']:.3f} (Dice: {best_dice['dice']:.4f})")
    
    return df

# Example of usage if predictions and targets are available
"""
# Assume all_preds and all_targets are lists of prediction and target tensors
threshold_df = analyze_threshold_sensitivity(all_preds, all_targets)
"""
```

## Model Comparison Analysis

```python
# Compare multiple models or runs
def compare_models(results_list, names):
    """Compare multiple models or different runs."""
    comparison = []
    
    for result, name in zip(results_list, names):
        comparison.append({
            'Model': name,
            'Dice': result.get('dice', 0),
            'IoU': result.get('iou', 0),
            'Precision': result.get('precision', 0),
            'Recall': result.get('recall', 0),
            'F1': result.get('f1', 0)
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Plot comparison
    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, data=comparison_df)
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# Example of usage with multiple result files
"""
# Load multiple result files
result1 = load_saved_results("../results/model1/evaluation_results.json")
result2 = load_saved_results("../results/model2/evaluation_results.json")
result3 = load_saved_results("../results/model3/evaluation_results.json")

# Compare models
comparison_df = compare_models(
    [result1, result2, result3],
    ['Baseline', 'Focal Loss', 'Ensemble']
)
"""
```

## Conclusion and Recommendations

```python
# Summarize findings and make recommendations

print("\n=== Model Performance Summary ===")
print(f"Overall Dice Score: {results.get('dice', 'N/A'):.4f}")
print(f"Overall IoU Score: {results.get('iou', 'N/A'):.4f}")

# Identify strengths and weaknesses
strengths = []
weaknesses = []

# Analyze based on results
if results.get('precision', 0) > 0.8:
    strengths.append("High precision (low false positive rate)")
else:
    weaknesses.append("Low precision (high false positive rate)")

if results.get('recall', 0) > 0.8:
    strengths.append("High recall (low false negative rate)")
else:
    weaknesses.append("Low recall (high false negative rate)")

if results.get('dice', 0) > 0.8:
    strengths.append("Good overall segmentation performance")
else:
    weaknesses.append("Moderate or poor segmentation performance")

print("\nStrengths:")
for s in strengths:
    print(f"- {s}")

print("\nWeaknesses:")
for w in weaknesses:
    print(f"- {w}")

print("\nRecommendations:")
# Make data-driven recommendations based on analysis
if results.get('precision', 0) < 0.8 and results.get('recall', 0) > 0.8:
    print("- Focus on reducing false positives")
elif results.get('precision', 0) > 0.8 and results.get('recall', 0) < 0.8:
    print("- Focus on reducing false negatives")
elif results.get('dice', 0) < 0.7:
    print("- Consider using a stronger backbone network")
    print("- Experiment with more extensive data augmentation")

if not strengths:
    print("- Revisit loss function choice and hyperparameters")
    print("- Consider ensemble methods to improve performance")
```

This notebook provides a comprehensive analysis of model performance, identifies strengths and weaknesses, and offers data-driven recommendations for improvement. Use the modular structure to add or remove sections as needed for your specific analysis needs.