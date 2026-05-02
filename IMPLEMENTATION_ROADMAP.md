# AI-Powered Deepfake Detection Project - Implementation Roadmap

## Project Overview
**Goal:** Build a dual-branch CNN system (spatial + frequency domain) for deepfake detection optimized for CPU inference.

**Constraints:** CPU-only, 4-8 weeks, balanced accuracy/speed

---

## Phase 1: Environment Setup & Data Preparation (Week 1)

### Week 1 Tasks
- [x] Create project structure
- [ ] Set up virtual environment
- [ ] Install dependencies (optimized for CPU)
- [ ] Download FaceForensics++ dataset
- [ ] Create data pipeline & preprocessing
- [ ] Implement face detection with MTCNN

### Key Libraries (CPU-Optimized)
```
TensorFlow/Keras (CPU inference faster than PyTorch)
OpenCV (face detection & preprocessing)
NumPy, SciPy (FFT operations)
Streamlit (UI)
Scikit-learn (metrics)
Pillow (image handling)
```

### Dataset Strategy
- **FaceForensics++**: ~500K videos, extract key frames
- **Alternative (if bandwidth limited)**: Start with smaller subset (10K-20K images)
- **Split Strategy**: 70% train / 10% validation / 20% test (stratified by manipulation type)

---

## Phase 2: Model Architecture Development (Week 2-3)

### Architecture Specifications

#### Spatial Branch
```
Input: RGB Image (224×224×3)
├─ Conv2D(32, 3×3) + ReLU
├─ MaxPool(2×2)
├─ Conv2D(64, 3×3) + ReLU
├─ MaxPool(2×2)
├─ Conv2D(128, 3×3) + ReLU
├─ GlobalAveragePooling
└─ Dense(256) + Dropout(0.5)
```

**Rationale for CPU:**
- Fewer parameters than MobileNetV2 (lighter)
- Faster training & inference (~100-200ms per image)
- Still captures spatial artifacts (blending, textures)

#### Frequency Branch
```
Input: FFT Magnitude Spectrum (224×224×1)
├─ Conv2D(16, 3×3) + ReLU
├─ MaxPool(2×2)
├─ Conv2D(32, 3×3) + ReLU
├─ GlobalAveragePooling
└─ Dense(128) + Dropout(0.5)
```

**Smaller than spatial branch** because frequency data is more discriminative

#### Fusion & Classification
```
Concatenate(spatial_features, frequency_features)
├─ Dense(256) + ReLU + Dropout(0.3)
├─ Dense(128) + ReLU + Dropout(0.3)
└─ Dense(1) + Sigmoid → Output (0=Real, 1=Fake)
```

---

## Phase 3: Training Pipeline (Week 3-4)

### Training Configuration (CPU-Optimized)
```
Optimizer: Adam (lr=1e-4, beta_1=0.9, beta_2=0.999)
Loss Function: Binary Crossentropy with class weights
Batch Size: 16 (CPU constraint)
Epochs: 50-100 (with early stopping)
Validation Split: 10%

Callbacks:
- Early Stopping (patience=10, monitor val_loss)
- Model Checkpoint (save best weights)
- ReduceLROnPlateau (lr *= 0.5 if no improvement)
```

### Data Augmentation
```python
- RandomFlip(horizontal=True)
- RandomRotation(degrees=10)
- RandomZoom(0.8-1.2)
- RandomBrightness(0.1)
- GaussianNoise(σ=0.01) - to simulate compression
- RandomContrast(0.1)
```

---

## Phase 4: Evaluation & Testing (Week 4-5)

### Evaluation Metrics
```
1. Accuracy: (TP+TN)/(TP+TN+FP+FN)
2. Precision: TP/(TP+FP) - False positive rate
3. Recall: TP/(TP+FN) - Detection rate
4. F1-Score: Harmonic mean of precision & recall
5. AUC-ROC: Model discrimination ability
6. Confusion Matrix: Visualization
7. Inference Time: ms per image (target: <500ms on CPU)
```

### Testing Strategy
- **Cross-Dataset Testing**: Train on FaceForensics++, test on Kaggle data
- **Robustness Testing**: Add compression, blur, noise, brightness changes
- **Ablation Study**: 
  - Spatial branch only
  - Frequency branch only
  - Combined (both branches)

---

## Phase 5: UI & Deployment (Week 5-6)

### Streamlit Application Features
```
1. Image Upload (JPG, PNG, PNG, BMP)
2. Real-time Detection
3. Confidence Score (0-1 scale)
4. Visualization:
   - Original image
   - Predicted class (Real/Fake)
   - Confidence bar
   - Processing time
5. Batch Processing (multiple images)
6. Model Info & Metrics Display
```

---

## Phase 6: Optimization & Documentation (Week 6-8)

### Optimization Tasks
- [ ] Quantize model (reduce size 50-80%)
- [ ] Convert to TF Lite (optional, for mobile)
- [ ] Profile inference time
- [ ] Optimize preprocessing pipeline
- [ ] Create inference cache

### Documentation
- [ ] README with setup instructions
- [ ] API documentation
- [ ] Dataset description & download links
- [ ] Training results & metrics
- [ ] Inference benchmarks
- [ ] Limitations & future work

---

## File Structure
```
deepfake-detection/
├── data/
│   ├── raw/                    # Original FaceForensics++ data
│   ├── processed/              # Preprocessed images
│   └── train_val_test_split/   # Split datasets
├── models/
│   ├── spatial_branch.h5
│   ├── frequency_branch.h5
│   └── combined_model.h5
├── src/
│   ├── preprocessing.py        # Face detection & FFT
│   ├── model_architecture.py   # CNN definitions
│   ├── training.py             # Training loop
│   ├── evaluation.py           # Metrics & testing
│   └── inference.py            # Prediction pipeline
├── ui/
│   └── streamlit_app.py        # Web interface
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_evaluation.ipynb
├── requirements.txt
├── config.yaml                 # Hyperparameters
├── README.md
└── results/
    ├── training_logs/
    ├── metrics/
    └── sample_predictions/
```

---

## CPU Optimization Tips

### 1. Reduce Model Complexity
✅ Small CNN instead of ResNet/Xception
✅ Batch size = 16 (instead of 32/64)
✅ Image size = 224×224 (not 256 or higher)

### 2. Efficient Preprocessing
✅ Vectorized NumPy operations (avoid loops)
✅ Cache preprocessed images to disk
✅ Use PIL instead of OpenCV where possible

### 3. Training Optimization
✅ Use tf.data.Dataset for pipeline efficiency
✅ Set num_parallel_calls for preprocessing
✅ Use mixed precision (not available on all CPUs, but try)

### 4. Inference Optimization
✅ Load model once, reuse for multiple predictions
✅ Use model.predict_on_batch() for batch processing
✅ Consider TensorFlow Lite quantization

---

## Success Criteria

### Must-Have (MVP)
- [ ] Model achieves >85% accuracy on test set
- [ ] Inference time < 500ms per image (CPU)
- [ ] Handles both real and deepfake images
- [ ] Working Streamlit UI
- [ ] Complete documentation

### Nice-to-Have (Polish)
- [ ] >90% accuracy
- [ ] <300ms inference time
- [ ] Batch processing capability
- [ ] Robustness to compression/noise
- [ ] Model comparison (spatial vs frequency vs combined)

---

## Week-by-Week Timeline

| Week | Goal | Deliverable |
|------|------|-------------|
| 1 | Setup + Data Prep | Preprocessed dataset, face detection pipeline |
| 2-3 | Model Development | Spatial + Frequency branches, fusion layer |
| 4 | Training | Trained model with validation metrics |
| 5 | Evaluation | Test results, ablation study, robustness tests |
| 6 | UI + Deployment | Working Streamlit app |
| 7-8 | Optimization + Polish | Documentation, optimization, final testing |

---

## Next Steps
1. Create project directory structure
2. Set up virtual environment
3. Install dependencies
4. Build preprocessing pipeline
5. Implement model architecture
6. Start training with sample data
