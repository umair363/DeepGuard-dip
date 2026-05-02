# 🎭 AI-Powered Deepfake Detection System

A dual-branch CNN system for detecting deepfake images using spatial and frequency domain analysis. Optimized for CPU inference.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Web Interface](#web-interface)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

This project implements a deepfake detection system that combines:

1. **Spatial Branch**: Analyzes RGB images for visual artifacts (blending issues, texture inconsistencies)
2. **Frequency Branch**: Analyzes FFT spectra for AI generation patterns
3. **Fusion Classifier**: Combines features from both branches for robust classification

**Key Innovation**: Dual-domain analysis provides complementary information that neither branch alone can capture.

## ✨ Features

- ✅ **Dual-branch CNN architecture** (Spatial + Frequency)
- ✅ **CPU-optimized** lightweight model (~2M parameters)
- ✅ **Data augmentation pipeline** for robust training
- ✅ **Comprehensive evaluation metrics** (Accuracy, Precision, Recall, F1, AUC-ROC)
- ✅ **Ablation study** to analyze component contribution
- ✅ **Robustness testing** against noise, blur, compression
- ✅ **Streamlit web interface** for easy inference
- ✅ **Batch processing** capability
- ✅ **Cross-dataset evaluation** (FaceForensics++ → Kaggle)

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB (for dataset and models)
- **CPU**: Intel/AMD processor with >2GHz

### Hardware Setup Tested
- ✅ CPU-only systems (Intel Core i5, i7)
- ✅ MacBook Pro (M1/M2)
- ✅ Linux workstations
- ✅ Google Colab (free tier)

### Dependencies
See `requirements.txt` for complete list. Key libraries:
- TensorFlow 2.14
- OpenCV
- NumPy, SciPy
- Streamlit
- Scikit-learn

## 🔧 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd deepfake-detection
```

### 2. Create Virtual Environment
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

## 📊 Dataset Preparation

### Option 1: Use FaceForensics++ (Recommended)
1. Visit: https://github.com/ondyari/FaceForensics
2. Download the dataset (requires registration)
3. Extract to `./data/raw/`

### Option 2: Use Kaggle Deepfake Challenge
1. Download from: https://www.kaggle.com/c/deepfake-detection-challenge
2. Extract frames: `ffmpeg -i video.mp4 frame_%04d.jpg`

### Option 3: Quick Start with Smaller Dataset
For testing purposes, you can use a subset:
```bash
# Create small test dataset
python -c "
import os
from pathlib import Path
Path('./data/raw/real').mkdir(parents=True, exist_ok=True)
Path('./data/raw/fake').mkdir(parents=True, exist_ok=True)
print('Created directory structure')
"
```

### Data Organization
```
data/
├── raw/
│   ├── real/          # Real face images
│   │   ├── img001.jpg
│   │   └── ...
│   └── fake/          # Deepfake images
│       ├── img001.jpg
│       └── ...
└── processed/         # Preprocessed images (created by training script)
    ├── spatial/
    └── frequency/
```

## 🚀 Training

### 1. Preprocess Data
```python
from src_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(target_size=(224, 224))

# Process real images
preprocessor.process_batch(
    './data/raw/real',
    './data/processed/real',
    detect_face=True
)

# Process fake images
preprocessor.process_batch(
    './data/raw/fake',
    './data/processed/fake',
    detect_face=True
)
```

### 2. Train Model
```bash
python train_model.py --epochs 100 --batch_size 16 --learning_rate 0.0001
```

**Expected Training Time**: 
- CPU with 10K images: ~4-6 hours
- CPU with 50K images: ~20-30 hours

**Training Configuration**:
- Optimizer: Adam (lr=1e-4)
- Loss: Binary Crossentropy
- Batch Size: 16 (CPU constraint)
- Early Stopping: Patience=10
- Learning Rate Schedule: ReduceLROnPlateau

### 3. Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir ./results/training_logs
```

## 📈 Evaluation

### Test Model on Test Set
```bash
python evaluate_model.py --model_path ./models/combined_model.h5
```

### Run Ablation Study
```python
from src_evaluation import AblationStudy

# Ablation shows contribution of each branch
results = ablation_study.run_ablation(X_spatial_test, X_frequency_test, y_test)
```

### Robustness Testing
```python
from src_evaluation import RobustnessTest

# Test against corruptions
robustness_results = RobustnessTest.test_robustness(
    model, X_spatial_test, X_frequency_test, y_test
)
```

## 🔮 Inference

### Single Image Prediction
```python
from src_preprocessing import DataPreprocessor
from src_model_architecture import DualBranchDeepfakeDetector
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('./models/combined_model.h5')

# Load and preprocess image
preprocessor = DataPreprocessor()
spatial, frequency = preprocessor.process_image('./test_image.jpg')

# Predict
confidence = model.predict([
    np.expand_dims(spatial, 0),
    np.expand_dims(frequency, 0)
])

print(f"Confidence: {confidence[0][0]:.4f}")
print(f"Prediction: {'FAKE' if confidence[0][0] > 0.5 else 'REAL'}")
```

### Batch Prediction
```python
# Process multiple images
predictions = model.predict([batch_spatial, batch_frequency])
for i, conf in enumerate(predictions):
    label = 'FAKE' if conf[0] > 0.5 else 'REAL'
    print(f"Image {i}: {label} ({conf[0]:.2%})")
```

## 🌐 Web Interface

### Start Streamlit App
```bash
streamlit run streamlit_app.py
```

**Features**:
- Upload image for detection
- Real-time predictions
- Confidence visualization
- Domain analysis display
- Adjustable confidence threshold
- Detailed interpretation

**Access**: Open http://localhost:8501 in your browser

## 📊 Results

### Expected Performance

| Metric | Spatial Only | Frequency Only | Combined |
|--------|-------------|----------------|----------|
| Accuracy | 85-88% | 82-86% | 88-92% |
| Precision | 0.84-0.87 | 0.81-0.85 | 0.87-0.91 |
| Recall | 0.86-0.89 | 0.83-0.87 | 0.89-0.93 |
| F1-Score | 0.85-0.88 | 0.82-0.86 | 0.88-0.92 |
| AUC-ROC | 0.90-0.93 | 0.88-0.91 | 0.92-0.95 |
| Inference Time | 150-200ms | 80-120ms | 230-280ms |

*Results vary based on dataset size and quality*

## 🐛 Troubleshooting

### Issue: "No face detected"
**Solution**: Ensure faces are clearly visible and frontal
```python
face_detector.detect_face(image, padding=20)  # Increase padding if needed
```

### Issue: Out of Memory
**Solution**: Reduce batch size
```python
# In config.yaml
batch_size: 8  # or lower
```

### Issue: Model not found
**Solution**: Ensure model is trained first
```bash
python train_model.py  # Train before inference
```

### Issue: Slow inference on CPU
**Expected behavior** for CPU. Use quantization to speed up:
```python
# Convert to TFLite for mobile/edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Issue: Poor accuracy
**Solutions**:
1. Increase dataset size (10K+ samples)
2. Tune hyperparameters (learning rate, batch size)
3. Increase epochs and enable early stopping
4. Use data augmentation more aggressively

## 📁 Project Structure

```
deepfake-detection/
├── src_preprocessing.py         # Data preprocessing
├── src_model_architecture.py    # CNN architecture
├── src_training.py              # Training pipeline
├── src_evaluation.py            # Evaluation metrics
├── train_model.py              # Main training script
├── evaluate_model.py           # Main evaluation script
├── streamlit_app.py            # Web interface
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── IMPLEMENTATION_ROADMAP.md   # Detailed timeline
│
├── data/
│   ├── raw/                    # Original images
│   ├── processed/              # Preprocessed images
│   └── train_val_test_split/   # Split datasets
│
├── models/
│   ├── combined_model.h5      # Full model
│   ├── spatial_branch.h5      # Spatial branch
│   └── frequency_branch.h5    # Frequency branch
│
├── results/
│   ├── training_logs/         # Training history
│   ├── metrics/               # Evaluation results
│   ├── sample_predictions/    # Example outputs
│   └── confusion_matrices/    # Visualizations
│
└── notebooks/
    ├── 01_eda.ipynb           # Exploratory analysis
    ├── 02_preprocessing.ipynb  # Data preparation
    └── 03_evaluation.ipynb    # Results analysis
```

## 🚀 Future Enhancements

### Short Term (Week 6-7)
- [ ] Video deepfake detection
- [ ] Multi-frame analysis
- [ ] Real-time webcam detection
- [ ] Model quantization for mobile

### Medium Term
- [ ] Attention mechanisms
- [ ] Transformer-based detector
- [ ] Larger dataset (500K+ images)
- [ ] Production deployment (AWS/GCP)

### Long Term
- [ ] GAN detection module
- [ ] Forensic analysis report
- [ ] Source tracing
- [ ] Defense mechanisms

## 📝 License

[Specify your license here]

## 👥 Contributors

- [Your Name] - Project Lead
- [Team Members] - Implementation

## 📚 References

1. **FaceForensics++**: Rössler et al. (2019)
2. **Frequency Analysis**: Frank et al. (2020)
3. **CNN Architecture**: He et al. (ResNet), Chollet (Xception)

## 🙏 Acknowledgments

- FaceForensics++ dataset creators
- Kaggle deepfake challenge community
- TensorFlow and open-source ML community

## 📧 Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: See IMPLEMENTATION_ROADMAP.md

---

**Last Updated**: April 2026
**Version**: 1.0.0
**Status**: Research/Development
