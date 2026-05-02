"""
Main Training Script for Deepfake Detection
Orchestrates: data loading, model creation, training, evaluation, and saving
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
from src_preprocessing import DataPreprocessor
from src_model_architecture import DualBranchDeepfakeDetector
from src_training import TrainingManager, DataLoader
from src_evaluation import MetricsCalculator, Visualizer, AblationStudy, RobustnessTest


class DeepfakeDetectionPipeline:
    """Complete training pipeline"""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.history = None
        
        logger.info("Pipeline initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['paths']['raw_data'],
            self.config['paths']['models_dir'],
            self.config['paths']['results_dir'],
            self.config['paths']['logs_dir'],
            self.config['paths']['metrics_dir'],
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directories created")
    
    def preprocess_data(self, input_dir, output_dir='./data/processed'):
        """Skipped: Using dynamic data generators for 140k dataset."""
        logger.info("Skipping offline preprocessing (using dynamic data generators).")
        pass
    
    def _get_files_and_labels(self, base_dir):
        """Helper to get file paths and labels from a directory (real=0, fake=1)"""
        files = []
        labels = []
        for class_name, label in [('real', 0), ('fake', 1)]:
            class_dir = os.path.join(base_dir, class_name)
            if os.path.exists(class_dir):
                for f in os.listdir(class_dir):
                    if f.endswith('.jpg') or f.endswith('.png'):
                        files.append(os.path.join(class_dir, f))
                        labels.append(label)
        return files, labels

    def load_and_split_data(self, processed_dir=None):
        """
        Create DataGenerators for train, validation, and test datasets.
        """
        logger.info("Initializing dynamic data generators...")
        from src_training import DataGenerator
        
        target_size = (self.config['model']['image_height'], self.config['model']['image_width'])
        batch_size = self.config['training']['batch_size']
        
        # Train
        train_files, train_labels = self._get_files_and_labels(self.config['paths']['train_data'])
        train_gen = DataGenerator(train_files, train_labels, batch_size=batch_size, target_size=target_size, shuffle=True)
        
        # Valid
        val_files, val_labels = self._get_files_and_labels(self.config['paths']['valid_data'])
        val_gen = DataGenerator(val_files, val_labels, batch_size=batch_size, target_size=target_size, shuffle=False)
        
        # Test
        test_files, test_labels = self._get_files_and_labels(self.config['paths']['test_data'])
        test_gen = DataGenerator(test_files, test_labels, batch_size=batch_size, target_size=target_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_files)}, Valid samples: {len(val_files)}, Test samples: {len(test_files)}")
        
        return {'train': train_gen, 'val': val_gen, 'test': test_gen}
    
    def build_model(self):
        """Build model architecture"""
        logger.info("Building model...")
        
        self.model = DualBranchDeepfakeDetector(
            spatial_input_shape=(
                self.config['model']['image_height'],
                self.config['model']['image_width'],
                3
            ),
            frequency_input_shape=(
                self.config['model']['image_height'],
                self.config['model']['image_width'],
                1
            )
        )
        
        self.model.build()
        
        # Compile
        self.model.compile(
            learning_rate=self.config['training']['learning_rate']
        )
        
        # Print summary
        logger.info("Model summary:")
        self.model.summary()
        
        # Count parameters
        params = self.model.count_parameters()
        logger.info(f"Total parameters: {params['total']:,}")
        
        self.model = self.model.get_model()
        return self.model
    
    def train(self, train_data, val_data, epochs=100):
        """
        Train model
        
        Args:
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            epochs: Number of epochs
        """
        logger.info("Starting training...")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array([0, 1]),
            y=np.array(train_data.labels)
        )
        class_weights = {i: w for i, w in enumerate(class_weights)}
        
        logger.info(f"Class weights: {class_weights}")
        
        # Create training manager
        manager = TrainingManager(
            self.model,
            output_dir=self.config['paths']['logs_dir']
        )
        
        # Train
        self.history = manager.train(
            train_generator=train_data,
            val_generator=val_data,
            epochs=epochs,
            class_weights=class_weights,
            patience=self.config['training']['early_stopping']['patience']
        )
        
        # Save history
        manager.save_history(
            os.path.join(self.config['paths']['results_dir'], 'training_history.json')
        )
        
        logger.info("Training complete")
        
        return self.history
    
    def evaluate(self, test_gen):
        """
        Evaluate model on test set using the data generator
        """
        logger.info("Evaluating on test set...")
        
        # Predict
        y_pred = self.model.predict(test_gen, verbose=1)
        y_test = np.array(test_gen.labels)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(y_test, y_pred)
        MetricsCalculator.print_metrics(metrics, dataset_name="Test")
        
        # Save metrics
        metrics_path = os.path.join(self.config['paths']['metrics_dir'], 'test_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Visualize
        Visualizer.plot_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(self.config['paths']['results_dir'], 'confusion_matrix.png')
        )
        
        Visualizer.plot_roc_curve(
            y_test, y_pred,
            save_path=os.path.join(self.config['paths']['results_dir'], 'roc_curve.png')
        )
        
        if self.history:
            Visualizer.plot_training_history(
                self.history,
                save_path=os.path.join(self.config['paths']['results_dir'], 'training_history.png')
            )
        
        return metrics
    
    def run_ablation_study(self, test_gen):
        """Run ablation study on a batch"""
        logger.info("Running ablation study...")
        X_batch, y_batch = test_gen[0]
        X_spatial, X_frequency = X_batch
        
        y_pred = self.model.predict([X_spatial, X_frequency], verbose=0)
        metrics = MetricsCalculator.calculate_metrics(y_batch, y_pred)
        logger.info("Combined model metrics (Ablation batch):")
        MetricsCalculator.print_metrics(metrics, dataset_name="Combined")
    
    def run_robustness_test(self, test_gen):
        """Test robustness against corruptions on a single batch to prevent OOM"""
        logger.info("Running robustness tests...")
        
        X_batch, y_batch = test_gen[0]
        X_spatial, X_frequency = X_batch
        
        results = RobustnessTest.test_robustness(
            self.model,
            X_spatial,
            X_frequency,
            y_batch,
            output_dir=self.config['paths']['results_dir']
        )
        return results
    
    def save_model(self):
        """Save trained model"""
        model_path = self.config['paths']['combined_model']
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def run_full_pipeline(self, raw_data_dir='./data/real_vs_fake/real-vs-fake', epochs=100):
        """
        Run complete training pipeline
        """
        logger.info("="*60)
        logger.info("DEEPFAKE DETECTION - FULL PIPELINE (DYNAMIC DATA GENERATOR)")
        logger.info("="*60)
        
        # Create directories
        self._create_directories()
        
        # Get data generators
        generators = self.load_and_split_data()
        
        # Build model
        self.build_model()
        
        # Train
        self.train(generators['train'], generators['val'], epochs=epochs)
        
        # Evaluate
        self.evaluate(generators['test'])
        
        # Ablation study
        self.run_ablation_study(generators['test'])
        
        # Robustness testing
        self.run_robustness_test(generators['test'])
        
        # Save model
        self.save_model()
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train deepfake detection model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--raw_data',
        type=str,
        default='./data/raw',
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline(config_path=args.config)
    
    # Run pipeline
    try:
        pipeline.run_full_pipeline(
            raw_data_dir=args.raw_data,
            epochs=args.epochs
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
