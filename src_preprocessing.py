"""
Preprocessing Module for Deepfake Detection
Handles: Face detection, image resizing, normalization, FFT processing
"""

import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
import os
from pathlib import Path
from tqdm import tqdm
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using OpenCV Haar Cascades (lightweight for CPU)"""
    
    def __init__(self, cascade_path=None, min_face_size=50):
        """
        Initialize face detector
        
        Args:
            cascade_path: Path to Haar Cascade XML file
            min_face_size: Minimum face size in pixels
        """
        if cascade_path is None:
            # Use OpenCV's default cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.min_face_size = min_face_size
        
        if self.face_cascade.empty():
            logger.warning("Failed to load cascade classifier")
    
    def detect_face(self, image, padding=20):
        """
        Detect face in image and return cropped region
        
        Args:
            image: Input image (numpy array or file path)
            padding: Padding around detected face (pixels)
        
        Returns:
            Cropped face image or None if no face detected
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Failed to read image: {image}")
                return None
        else:
            img = image.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        if len(faces) == 0:
            logger.warning("No face detected in image")
            return None
        
        # Get largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        cropped_face = img[y:y+h, x:x+w]
        
        return cropped_face
    
    def detect_faces_batch(self, image_dir, output_dir, padding=20):
        """
        Detect faces in batch of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save cropped faces
            padding: Padding around detected faces
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(image_dir) 
                 if f.lower().endswith(image_extensions)]
        
        successful = 0
        failed = 0
        
        for image_file in tqdm(images, desc="Detecting faces"):
            image_path = os.path.join(image_dir, image_file)
            
            try:
                cropped = self.detect_face(image_path, padding)
                
                if cropped is not None:
                    output_path = os.path.join(output_dir, image_file)
                    cv2.imwrite(output_path, cropped)
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"No face in {image_file}")
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {image_file}: {str(e)}")
        
        logger.info(f"Batch processing complete. Successful: {successful}, Failed: {failed}")


class ImagePreprocessor:
    """Image preprocessing: resizing, normalization"""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def resize_image(self, image):
        """Resize image to target size using bilinear interpolation"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to read image: {image}")
        
        # Ensure we have BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        resized = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize(self, image):
        """Normalize pixel values to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def preprocess_spatial(self, image):
        """
        Preprocess for spatial branch
        
        Returns: Normalized RGB image (224x224x3)
        """
        resized = self.resize_image(image)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        normalized = self.normalize(rgb)
        
        return normalized
    
    def preprocess_frequency(self, image):
        """
        Preprocess for frequency branch
        
        Returns: Log-scaled FFT magnitude spectrum (224x224x1)
        """
        resized = self.resize_image(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Compute 2D FFT
        fft_result = fft2(gray)
        fft_shift = fftshift(fft_result)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shift)
        
        # Log scale to compress dynamic range
        log_magnitude = np.log1p(magnitude)
        
        # Normalize to [0, 1]
        log_magnitude = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)
        
        # Add channel dimension
        log_magnitude = np.expand_dims(log_magnitude, axis=-1)
        
        return log_magnitude.astype(np.float32)


class DataPreprocessor:
    """Combined preprocessing pipeline"""
    
    def __init__(self, target_size=(224, 224), min_face_size=50):
        """
        Initialize data preprocessor
        
        Args:
            target_size: Target image size
            min_face_size: Minimum face size for detection
        """
        self.face_detector = FaceDetector(min_face_size=min_face_size)
        self.image_preprocessor = ImagePreprocessor(target_size)
    
    def process_image(self, image_path, detect_face=True):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to input image
            detect_face: Whether to detect and crop face
        
        Returns:
            Tuple of (spatial_branch_input, frequency_branch_input)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read: {image_path}")
                return None, None
            
            # Detect face if requested
            if detect_face:
                image = self.face_detector.detect_face(image, padding=20)
                if image is None:
                    return None, None
            
            # Preprocess for spatial branch
            spatial_input = self.image_preprocessor.preprocess_spatial(image)
            
            # Preprocess for frequency branch
            frequency_input = self.image_preprocessor.preprocess_frequency(image)
            
            return spatial_input, frequency_input
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None, None
    
    def process_batch(self, image_dir, output_dir, detect_face=True):
        """
        Process batch of images
        
        Args:
            image_dir: Input image directory
            output_dir: Output directory for processed images
            detect_face: Whether to detect faces
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        spatial_dir = os.path.join(output_dir, 'spatial')
        frequency_dir = os.path.join(output_dir, 'frequency')
        
        Path(spatial_dir).mkdir(exist_ok=True)
        Path(frequency_dir).mkdir(exist_ok=True)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(image_dir) 
                 if f.lower().endswith(image_extensions)]
        
        successful = 0
        failed = 0
        
        for image_file in tqdm(images, desc="Preprocessing images"):
            image_path = os.path.join(image_dir, image_file)
            
            spatial, frequency = self.process_image(image_path, detect_face)
            
            if spatial is not None and frequency is not None:
                # Save as numpy arrays
                spatial_path = os.path.join(spatial_dir, image_file.replace('.jpg', '.npy').replace('.png', '.npy'))
                frequency_path = os.path.join(frequency_dir, image_file.replace('.jpg', '.npy').replace('.png', '.npy'))
                
                np.save(spatial_path, spatial)
                np.save(frequency_path, frequency)
                
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Batch preprocessing complete. Successful: {successful}, Failed: {failed}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Process a single image
    preprocessor = DataPreprocessor(target_size=(224, 224))
    
    # Test with sample image (create a dummy one for testing)
    print("Data preprocessing module loaded successfully!")
    print("Functions available:")
    print("  - FaceDetector: Detect and crop faces")
    print("  - ImagePreprocessor: Resize and normalize images")
    print("  - DataPreprocessor: Complete pipeline")
