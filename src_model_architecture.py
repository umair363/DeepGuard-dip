"""
Model Architecture for Deepfake Detection
Implements dual-branch CNN: Spatial + Frequency domain analysis
Optimized for CPU inference
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import logging

logger = logging.getLogger(__name__)


class SpatialBranch:
    """
    Spatial Branch: Analyzes RGB image for visual artifacts
    - Detects unnatural blending, weird textures, skin tone issues
    - Lightweight CNN optimized for CPU
    """
    
    @staticmethod
    def build(input_shape=(224, 224, 3)):
        """
        Build spatial branch model using MobileNetV2 for powerful feature extraction.
        
        Args:
            input_shape: Input image shape (height, width, channels)
        
        Returns:
            Keras model
        """
        # Load pre-trained MobileNetV2 without its top classification layer
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Fine-tune the backbone
        base_model.trainable = True

        model = models.Sequential([
            # Input
            layers.Input(shape=input_shape),
            
            # Data Augmentation (active only during training)
            layers.RandomFlip("horizontal", name='aug_flip'),
            layers.RandomRotation(0.1, name='aug_rot'),
            layers.RandomZoom(0.1, name='aug_zoom'),
            
            # Map [0, 1] input to [-1, 1] for MobileNetV2
            layers.Rescaling(scale=2.0, offset=-1.0, name='mobilenet_preprocess'),
            
            # Backbone
            base_model,
            
            # Global pooling
            layers.GlobalAveragePooling2D(name='spatial_global_pool'),
            
            # Dense layer
            layers.Dense(256, activation='relu', name='spatial_dense1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
        ], name='spatial_branch')
        
        return model


class FrequencyBranch:
    """
    Frequency Branch: Analyzes FFT spectrum for hidden patterns
    - Detects anomalies in frequency domain left by AI generation tools
    - Smaller network (frequency data is more discriminative)
    """
    
    @staticmethod
    def build(input_shape=(224, 224, 1)):
        """
        Build frequency branch model
        
        Args:
            input_shape: FFT spectrum shape (height, width, channels)
        
        Returns:
            Keras model
        """
        model = models.Sequential([
            # Input
            layers.Input(shape=input_shape),
            
            # Block 1: 16 filters
            layers.Conv2D(16, kernel_size=3, padding='same', activation='relu',
                         name='freq_conv1'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2, name='freq_pool1'),
            
            # Block 2: 32 filters
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu',
                         name='freq_conv2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2, name='freq_pool2'),
            
            # Global pooling
            layers.GlobalAveragePooling2D(name='freq_global_pool'),
            
            # Dense layer (smaller than spatial branch)
            layers.Dense(128, activation='relu', name='freq_dense1'),
            layers.Dropout(0.5),
        ], name='frequency_branch')
        
        return model


class FusionClassifier:
    """
    Fusion Classifier: Combines spatial and frequency features
    - Concatenates features from both branches
    - Applies classification layers
    - Outputs probability (0=Real, 1=Fake)
    """
    
    @staticmethod
    def build(spatial_features=256, frequency_features=128):
        """
        Build fusion and classification layers
        
        Args:
            spatial_features: Output size of spatial branch
            frequency_features: Output size of frequency branch
        
        Returns:
            Keras model (functional API)
        """
        # Input layers for features from both branches
        spatial_input = layers.Input(shape=(spatial_features,), name='spatial_features')
        frequency_input = layers.Input(shape=(frequency_features,), name='frequency_features')
        
        # Concatenate features
        concat = layers.Concatenate(name='feature_concat')([spatial_input, frequency_input])
        
        # Dense layer 1
        dense1 = layers.Dense(256, activation='relu', name='fusion_dense1')(concat)
        dense1 = layers.Dropout(0.3)(dense1)
        
        # Dense layer 2
        dense2 = layers.Dense(128, activation='relu', name='fusion_dense2')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(dense2)
        
        # Create model
        model = models.Model(inputs=[spatial_input, frequency_input], outputs=output,
                            name='fusion_classifier')
        
        return model


class DualBranchDeepfakeDetector:
    """
    Complete Dual-Branch Model for Deepfake Detection
    Combines spatial and frequency domain analysis
    """
    
    def __init__(self, spatial_input_shape=(224, 224, 3),
                 frequency_input_shape=(224, 224, 1)):
        """
        Initialize model architecture
        
        Args:
            spatial_input_shape: Input shape for spatial branch
            frequency_input_shape: Input shape for frequency branch
        """
        self.spatial_input_shape = spatial_input_shape
        self.frequency_input_shape = frequency_input_shape
        self.model = None
    
    def build(self):
        """
        Build the complete dual-branch model
        
        Returns:
            Compiled Keras model
        """
        # Build branches
        spatial_branch = SpatialBranch.build(self.spatial_input_shape)
        frequency_branch = FrequencyBranch.build(self.frequency_input_shape)
        
        # Input layers
        spatial_input = layers.Input(shape=self.spatial_input_shape, name='spatial_input')
        frequency_input = layers.Input(shape=self.frequency_input_shape, name='frequency_input')
        
        # Extract features from branches
        spatial_features = spatial_branch(spatial_input)
        frequency_features = frequency_branch(frequency_input)
        
        # Fusion and classification
        fusion_classifier = FusionClassifier.build(
            spatial_features=spatial_features.shape[-1],
            frequency_features=frequency_features.shape[-1]
        )
        
        output = fusion_classifier([spatial_features, frequency_features])
        
        # Create complete model
        self.model = models.Model(
            inputs=[spatial_input, frequency_input],
            outputs=output,
            name='dual_branch_deepfake_detector'
        )
        
        return self.model
    
    def compile(self, learning_rate=1e-4, loss='binary_crossentropy',
                metrics=None):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            self.build()
        
        if metrics is None:
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
    
    def get_model(self):
        """Get the built model"""
        if self.model is None:
            self.build()
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build()
        self.model.summary()
    
    def count_parameters(self):
        """Count total trainable parameters"""
        if self.model is None:
            self.build()
        
        trainable = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        non_trainable = sum(tf.size(w).numpy() for w in self.model.non_trainable_weights)
        
        return {
            'trainable': int(trainable),
            'non_trainable': int(non_trainable),
            'total': int(trainable + non_trainable)
        }
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model from file"""
        model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


# Model factory function
def create_deepfake_detector(spatial_shape=(224, 224, 3),
                            frequency_shape=(224, 224, 1),
                            compile=True,
                            learning_rate=1e-4):
    """
    Create and optionally compile a deepfake detector model
    
    Args:
        spatial_shape: Spatial input shape
        frequency_shape: Frequency input shape
        compile: Whether to compile the model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled/uncompiled Keras model
    """
    detector = DualBranchDeepfakeDetector(spatial_shape, frequency_shape)
    detector.build()
    
    if compile:
        detector.compile(learning_rate=learning_rate)
    
    return detector.get_model()


# Example usage and testing
if __name__ == "__main__":
    print("Creating deepfake detection model...")
    
    # Create model
    detector = DualBranchDeepfakeDetector()
    detector.build()
    detector.compile()
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    detector.summary()
    
    # Count parameters
    params = detector.count_parameters()
    print("\n" + "="*60)
    print("PARAMETER COUNT")
    print("="*60)
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Total parameters: {params['total']:,}")
    print("\nNote: This is CPU-optimized, relatively small compared to ResNet/Xception")
