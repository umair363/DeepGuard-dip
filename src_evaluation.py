"""
Evaluation Module for Deepfake Detection
Handles: Metrics calculation, visualization, ablation studies, robustness testing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (0-1)
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int).flatten()
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_true, y_pred_proba)),
            'true_negatives': int(confusion_matrix(y_true, y_pred)[0, 0]),
            'false_positives': int(confusion_matrix(y_true, y_pred)[0, 1]),
            'false_negatives': int(confusion_matrix(y_true, y_pred)[1, 0]),
            'true_positives': int(confusion_matrix(y_true, y_pred)[1, 1]),
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, dataset_name="Test"):
        """Print metrics in readable format"""
        print(f"\n{'='*50}")
        print(f"Evaluation Metrics - {dataset_name} Set")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"{'='*50}\n")


class Visualizer:
    """Visualization utilities for model evaluation"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5, 
                             title="Confusion Matrix", save_path=None):
        """Plot confusion matrix"""
        y_pred = (y_pred_proba >= threshold).astype(int).flatten()
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", save_path=None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve",
                                   save_path=None):
        """Plot precision-recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training history (loss and accuracy)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss', lw=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', lw=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Train Accuracy', lw=2)
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', lw=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


class AblationStudy:
    """Ablation study: test individual components"""
    
    def __init__(self, model, spatial_model, frequency_model):
        """
        Initialize ablation study
        
        Args:
            model: Complete dual-branch model
            spatial_model: Spatial branch model
            frequency_model: Frequency branch model
        """
        self.model = model
        self.spatial_model = spatial_model
        self.frequency_model = frequency_model
    
    def run_ablation(self, X_spatial, X_frequency, y_test):
        """
        Run ablation study on test set
        
        Args:
            X_spatial: Spatial test images
            X_frequency: Frequency test images
            y_test: Test labels
        
        Returns:
            Dictionary of ablation results
        """
        results = {}
        
        # Test 1: Combined model (baseline)
        y_pred_combined = self.model.predict([X_spatial, X_frequency], verbose=0)
        metrics_combined = MetricsCalculator.calculate_metrics(y_test, y_pred_combined)
        results['combined'] = metrics_combined
        
        logger.info("Ablation Study: Combined Model")
        MetricsCalculator.print_metrics(metrics_combined, dataset_name="Combined")
        
        # Test 2: Spatial branch only
        # Extract spatial features and pass through fusion classifier
        # For simplicity, we can evaluate just the spatial accuracy
        y_pred_spatial = self.spatial_model.predict(X_spatial, verbose=0)
        metrics_spatial = MetricsCalculator.calculate_metrics(y_test, y_pred_spatial)
        results['spatial_only'] = metrics_spatial
        
        logger.info("Ablation Study: Spatial Branch Only")
        MetricsCalculator.print_metrics(metrics_spatial, dataset_name="Spatial Only")
        
        # Test 3: Frequency branch only
        y_pred_frequency = self.frequency_model.predict(X_frequency, verbose=0)
        metrics_frequency = MetricsCalculator.calculate_metrics(y_test, y_pred_frequency)
        results['frequency_only'] = metrics_frequency
        
        logger.info("Ablation Study: Frequency Branch Only")
        MetricsCalculator.print_metrics(metrics_frequency, dataset_name="Frequency Only")
        
        # Summary
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        print(f"Combined:      F1={results['combined']['f1_score']:.4f}, AUC={results['combined']['auc_roc']:.4f}")
        print(f"Spatial Only:  F1={results['spatial_only']['f1_score']:.4f}, AUC={results['spatial_only']['auc_roc']:.4f}")
        print(f"Frequency Only: F1={results['frequency_only']['f1_score']:.4f}, AUC={results['frequency_only']['auc_roc']:.4f}")
        print("="*60 + "\n")
        
        return results


class RobustnessTest:
    """Test model robustness against corruptions"""
    
    @staticmethod
    def add_gaussian_noise(image, sigma=0.01):
        """Add Gaussian noise"""
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def add_compression_artifact(image, quality=80):
        """Simulate JPEG compression"""
        from PIL import Image as PILImage
        import io
        
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Convert BGR to RGB if 3 channels
        if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
            # Already in correct format
            pil_img = PILImage.fromarray(img_uint8)
        else:
            pil_img = PILImage.fromarray(img_uint8)
        
        # Save and reload with compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = PILImage.open(buffer)
        
        # Convert back
        return np.array(compressed_img).astype(np.float32) / 255.0
    
    @staticmethod
    def add_gaussian_blur(image, kernel_size=5):
        """Add Gaussian blur"""
        import cv2
        if len(image.shape) == 2:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            blurred = image.copy()
            for i in range(image.shape[2]):
                blurred[:,:,i] = cv2.GaussianBlur(image[:,:,i], (kernel_size, kernel_size), 0)
            return blurred
    
    @staticmethod
    def add_brightness_change(image, delta=0.1):
        """Change brightness"""
        return np.clip(image + delta, 0, 1)
    
    @classmethod
    def test_robustness(cls, model, X_spatial, X_frequency, y_test, 
                       output_dir=None):
        """
        Test model robustness
        
        Args:
            model: Model to test
            X_spatial: Spatial images
            X_frequency: Frequency images
            y_test: Test labels
            output_dir: Directory to save results
        
        Returns:
            Dictionary of robustness test results
        """
        results = {}
        
        # Original
        y_pred = model.predict([X_spatial, X_frequency], verbose=0)
        results['original'] = MetricsCalculator.calculate_metrics(y_test, y_pred)
        logger.info(f"Original: F1={results['original']['f1_score']:.4f}")
        
        # Gaussian Noise
        X_spatial_noise = np.array([cls.add_gaussian_noise(x, sigma=0.01) for x in X_spatial])
        X_frequency_noise = np.array([cls.add_gaussian_noise(x, sigma=0.01) for x in X_frequency])
        y_pred = model.predict([X_spatial_noise, X_frequency_noise], verbose=0)
        results['noise'] = MetricsCalculator.calculate_metrics(y_test, y_pred)
        logger.info(f"Gaussian Noise: F1={results['noise']['f1_score']:.4f}")
        
        # Gaussian Blur
        X_spatial_blur = np.array([cls.add_gaussian_blur(x, kernel_size=5) for x in X_spatial])
        X_frequency_blur = np.array([cls.add_gaussian_blur(x, kernel_size=5) for x in X_frequency])
        y_pred = model.predict([X_spatial_blur, X_frequency_blur], verbose=0)
        results['blur'] = MetricsCalculator.calculate_metrics(y_test, y_pred)
        logger.info(f"Gaussian Blur: F1={results['blur']['f1_score']:.4f}")
        
        # Brightness Change
        X_spatial_bright = np.array([cls.add_brightness_change(x, delta=0.1) for x in X_spatial])
        X_frequency_bright = np.array([cls.add_brightness_change(x, delta=0.1) for x in X_frequency])
        y_pred = model.predict([X_spatial_bright, X_frequency_bright], verbose=0)
        results['brightness'] = MetricsCalculator.calculate_metrics(y_test, y_pred)
        logger.info(f"Brightness Change: F1={results['brightness']['f1_score']:.4f}")
        
        # Summary
        print("\n" + "="*60)
        print("ROBUSTNESS TEST SUMMARY")
        print("="*60)
        for corruption, metrics in results.items():
            print(f"{corruption.capitalize():15s}: F1={metrics['f1_score']:.4f}, AUC={metrics['auc_roc']:.4f}")
        print("="*60 + "\n")
        
        return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Classes available:")
    print("  - MetricsCalculator: Calculate metrics")
    print("  - Visualizer: Plot results")
    print("  - AblationStudy: Component analysis")
    print("  - RobustnessTest: Corruption testing")
