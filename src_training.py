"""
Training Module for Deepfake Detection
Handles: Data loading, training loop, validation, callbacks, early stopping
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


import cv2
from scipy.fftpack import fft2, fftshift

class DataGenerator(keras.utils.Sequence):
    """Dynamic data generator that reads raw images and computes features on-the-fly"""
    
    def __init__(self, file_paths, labels, batch_size=16, target_size=(224, 224), shuffle=True):
        """
        Initialize data generator
        
        Args:
            file_paths: List of absolute or relative paths to .jpg files
            labels: List of integer labels corresponding to file_paths
            batch_size: Batch size
            target_size: Tuple for image resizing (height, width)
            shuffle: Whether to shuffle data after each epoch
        """
        super().__init__()
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
        
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def _process_image(self, filepath):
        """Read image, resize, and compute spatial and frequency outputs"""
        # Load image with cv2
        img = cv2.imread(filepath)
        if img is None:
            # Fallback for corrupted images: create dummy zeros
            spatial = np.zeros((*self.target_size, 3), dtype=np.float32)
            frequency = np.zeros((*self.target_size, 1), dtype=np.float32)
            return spatial, frequency
            
        # Resize
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Spatial: Convert BGR to RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        spatial = img_rgb.astype(np.float32) / 255.0
        
        # Frequency: Convert to grayscale and compute FFT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        fft_result = fft2(gray)
        fft_shift = fftshift(fft_result)
        magnitude = np.abs(fft_shift)
        log_magnitude = np.log1p(magnitude)
        
        # Normalize frequency spectrum
        freq_min = log_magnitude.min()
        freq_max = log_magnitude.max()
        if freq_max > freq_min:
            log_magnitude = (log_magnitude - freq_min) / (freq_max - freq_min)
            
        frequency = np.expand_dims(log_magnitude, axis=-1).astype(np.float32)
        
        return spatial, frequency

    def __getitem__(self, index):
        """Get one batch of data"""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_spatial = []
        batch_frequency = []
        batch_labels = []
        
        for i in batch_indices:
            spatial, freq = self._process_image(self.file_paths[i])
            batch_spatial.append(spatial)
            batch_frequency.append(freq)
            batch_labels.append(self.labels[i])
            
        return (np.array(batch_spatial), np.array(batch_frequency)), np.array(batch_labels)


class TrainingManager:
    """Manages training process with callbacks and monitoring"""
    
    def __init__(self, model, output_dir='./results/training_logs'):
        """
        Initialize training manager
        
        Args:
            model: Keras model
            output_dir: Directory to save logs and checkpoints
        """
        self.model = model
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.best_val_loss = float('inf')
    
    def get_callbacks(self, patience=10):
        """
        Get training callbacks
        
        Args:
            patience: Early stopping patience
        
        Returns:
            List of callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'best_model_{timestamp}.h5')
        model_checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        )
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # TensorBoard
        tb_log_dir = os.path.join(self.output_dir, f'tensorboard_{timestamp}')
        tensorboard = callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=0,
            write_graph=True
        )
        
        return [model_checkpoint, early_stopping, reduce_lr, tensorboard]
    
    def train(self, train_generator, val_generator, epochs=100,
              class_weights=None, patience=10):
        """
        Train the model using data generators
        
        Args:
            train_generator: DataGenerator instance for training
            val_generator: DataGenerator instance for validation
            epochs: Number of epochs
            class_weights: Dictionary of class weights
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        cbs = self.get_callbacks(patience=patience)
        
        logger.info(f"Epochs: {epochs}")
        
        # Train
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=cbs,
            verbose=1
        )
        
        return self.history
    
    def save_history(self, filepath=None):
        """Save training history to JSON"""
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'training_history.json')
        
        history_dict = {
            'loss': [float(x) for x in self.history.history.get('loss', [])],
            'val_loss': [float(x) for x in self.history.history.get('val_loss', [])],
            'accuracy': [float(x) for x in self.history.history.get('accuracy', [])],
            'val_accuracy': [float(x) for x in self.history.history.get('val_accuracy', [])],
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        logger.info(f"History saved to {filepath}")


class DataLoader:
    """Utility for loading and splitting data"""
    
    @staticmethod
    def load_preprocessed_data(spatial_dir, frequency_dir, labels_dict):
        """
        Load preprocessed spatial and frequency images
        
        Args:
            spatial_dir: Directory with spatial images
            frequency_dir: Directory with frequency images
            labels_dict: Dictionary mapping image names to labels
        
        Returns:
            Tuple of (spatial_images, frequency_images, labels)
        """
        spatial_images = []
        frequency_images = []
        labels = []
        
        for image_name, label in labels_dict.items():
            spatial_path = os.path.join(spatial_dir, image_name.replace('.jpg', '.npy').replace('.png', '.npy'))
            frequency_path = os.path.join(frequency_dir, image_name.replace('.jpg', '.npy').replace('.png', '.npy'))
            
            if os.path.exists(spatial_path) and os.path.exists(frequency_path):
                try:
                    spatial = np.load(spatial_path)
                    frequency = np.load(frequency_path)
                    
                    spatial_images.append(spatial)
                    frequency_images.append(frequency)
                    labels.append(label)
                except Exception as e:
                    logger.error(f"Error loading {image_name}: {str(e)}")
        
        spatial_images = np.array(spatial_images)
        frequency_images = np.array(frequency_images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(labels)} samples")
        logger.info(f"Spatial shape: {spatial_images.shape}")
        logger.info(f"Frequency shape: {frequency_images.shape}")
        
        return spatial_images, frequency_images, labels
    
    @staticmethod
    def create_train_val_test_split(spatial_images, frequency_images, labels,
                                   train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                                   random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            spatial_images: Spatial input data
            frequency_images: Frequency input data
            labels: Labels
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            random_state: Random seed
        
        Returns:
            Dictionary with train, val, test splits
        """
        from sklearn.model_selection import train_test_split
        
        np.random.seed(random_state)
        
        # First split: train+val vs test
        X_spatial_temp, X_spatial_test, X_freq_temp, X_freq_test, y_temp, y_test = train_test_split(
            spatial_images, frequency_images, labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_spatial_train, X_spatial_val, X_freq_train, X_freq_val, y_train, y_val = train_test_split(
            X_spatial_temp, X_freq_temp, y_temp,
            test_size=val_ratio_adjusted,
            stratify=y_temp,
            random_state=random_state
        )
        
        splits = {
            'train': {
                'spatial': X_spatial_train,
                'frequency': X_freq_train,
                'labels': y_train
            },
            'val': {
                'spatial': X_spatial_val,
                'frequency': X_freq_val,
                'labels': y_val
            },
            'test': {
                'spatial': X_spatial_test,
                'frequency': X_freq_test,
                'labels': y_test
            }
        }
        
        logger.info(f"Train: {len(y_train)} samples")
        logger.info(f"Val: {len(y_val)} samples")
        logger.info(f"Test: {len(y_test)} samples")
        
        return splits


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Classes available:")
    print("  - DataGenerator: Batch data generation with augmentation")
    print("  - TrainingManager: Handle training with callbacks")
    print("  - DataLoader: Load and split data")
