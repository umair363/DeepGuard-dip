# 🚀 Quick Start Guide - Deepfake Detection

Complete setup and training in **4 steps**!

## Step 1: Setup (5 minutes)

```bash
# Clone/extract project
cd deepfake-detection

# Create virtual environment
python -m venv venv

# Activate
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

✅ Setup complete!

---

## Step 2: Prepare Data (varies)

### Option A: Quick Test (No data download needed)
```bash
# Create dummy dataset structure for testing
mkdir -p data/raw/{real,fake}
mkdir -p data/processed/{real,fake}/{spatial,frequency}

# Create a few sample images for testing
# (Copy some face images to data/raw/real and data/raw/fake)
```

### Option B: Use Real Dataset
```bash
# Download FaceForensics++ or Kaggle Deepfake dataset
# Extract to: data/raw/
#   ├── real/
#   │   ├── img_001.jpg
#   │   └── ...
#   └── fake/
#       ├── img_001.jpg
#       └── ...
```

---

## Step 3: Train Model (2-4 hours for 10K images)

### Basic Training
```bash
python train_model.py \
    --raw_data ./data/raw \
    --epochs 50 \
    --batch_size 16
```

### Advanced Options
```bash
python train_model.py \
    --config config.yaml \
    --raw_data ./data/raw \
    --epochs 100 \
    --batch_size 16
```

**What happens:**
1. ✓ Detects faces in images
2. ✓ Preprocesses for spatial + frequency domains
3. ✓ Builds dual-branch CNN model
4. ✓ Trains with early stopping
5. ✓ Evaluates on test set
6. ✓ Saves model to `./models/combined_model.h5`
7. ✓ Generates evaluation reports

**Monitor progress:**
```bash
# In another terminal
tensorboard --logdir ./results/training_logs
# Open http://localhost:6006
```

---

## Step 4: Run Inference (Choose one)

### Option A: Web Interface (Recommended!)
```bash
streamlit run streamlit_app.py
```
- Open http://localhost:8501
- Upload an image
- See predictions in real-time!

### Option B: Python Script
```python
import tensorflow as tf
import numpy as np
from src_preprocessing import DataPreprocessor

# Load model
model = tf.keras.models.load_model('./models/combined_model.h5')

# Load preprocessor
preprocessor = DataPreprocessor()

# Process image
spatial, frequency = preprocessor.process_image('./test_image.jpg')

# Predict
confidence = model.predict([
    np.expand_dims(spatial, 0),
    np.expand_dims(frequency, 0)
])

print(f"Confidence: {confidence[0][0]:.2%}")
print(f"Result: {'FAKE 🚨' if confidence[0][0] > 0.5 else 'REAL ✓'}")
```

### Option C: Batch Processing
```python
import glob
import numpy as np

# Get all test images
test_images = glob.glob('./data/raw/fake/*.jpg')[:10]

for img_path in test_images:
    spatial, frequency = preprocessor.process_image(img_path)
    confidence = model.predict([
        np.expand_dims(spatial, 0),
        np.expand_dims(frequency, 0)
    ])[0][0]
    
    print(f"{img_path}: {confidence:.2%} fake")
```

---

## 📊 Expected Results

After training on 10K-50K images:

| Metric | Value |
|--------|-------|
| Accuracy | 88-92% |
| Precision | 0.87-0.91 |
| Recall | 0.89-0.93 |
| F1-Score | 0.88-0.92 |
| AUC-ROC | 0.92-0.95 |
| Training Time (CPU) | 4-6 hours |
| Inference Time | 200-300ms |

---

## 🐛 Common Issues & Fixes

### "No module named 'src_preprocessing'"
```bash
# Make sure you're in the project root directory
pwd  # Should be: .../deepfake-detection
ls   # Should show: src_preprocessing.py, train_model.py, etc
```

### "CUDA not found" (This is OK!)
```bash
# TensorFlow will automatically use CPU
# This is expected and normal for CPU-only systems
```

### Out of Memory
```bash
# In config.yaml, reduce batch size:
batch_size: 8  # instead of 16
```

### "Model not found" during inference
```bash
# First train the model:
python train_model.py --epochs 50
# Model will be saved to ./models/combined_model.h5
```

### Slow preprocessing
```bash
# First run, preprocessing takes time. Subsequent runs will be faster.
# Results are cached in ./data/processed/
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `train_model.py` | Main training script |
| `streamlit_app.py` | Web interface |
| `config.yaml` | All hyperparameters |
| `src_preprocessing.py` | Image preprocessing |
| `src_model_architecture.py` | Model definition |
| `src_training.py` | Training utilities |
| `src_evaluation.py` | Metrics & visualization |
| `./models/combined_model.h5` | Trained model (after training) |

---

## 🎯 Next Steps After Training

1. **Evaluate Results**
   ```bash
   # Check ./results/ for:
   # - confusion_matrix.png
   # - roc_curve.png
   # - training_history.json
   ```

2. **Run Ablation Study**
   ```bash
   # Edit train_model.py to uncomment ablation study section
   python train_model.py --epochs 100
   ```

3. **Test Robustness**
   ```python
   from src_evaluation import RobustnessTest
   # See results in ./results/robustness_test/
   ```

4. **Deploy Model**
   ```bash
   # Option 1: Keep using Streamlit
   streamlit run streamlit_app.py
   
   # Option 2: Convert to TFLite (for mobile)
   # See advanced section in README.md
   ```

---

## 💡 Tips for Better Results

1. **More Data = Better Model**
   - 1K images: Quick test
   - 10K images: Good baseline
   - 50K+ images: Production quality

2. **Balanced Dataset**
   - Equal real and fake images
   - Check: `./results/metrics/test_metrics.json`

3. **Tune Hyperparameters**
   - Edit `config.yaml`
   - Retrain with `python train_model.py`

4. **Monitor Training**
   - Use TensorBoard
   - Check early stopping (patience=10)

5. **Test Cross-Dataset**
   - Train on FaceForensics++
   - Test on Kaggle dataset

---

## 📞 Getting Help

1. **Check README.md** for detailed documentation
2. **Review IMPLEMENTATION_ROADMAP.md** for architecture details
3. **Check logs** in `./results/training_logs/`
4. **Enable debug mode**: `logging.getLogger().setLevel(logging.DEBUG)`

---

## ⏱️ Timeline Estimates

| Task | CPU Time | GPU Time |
|------|----------|----------|
| Setup | 10 min | 10 min |
| Data Prep (10K images) | 30 min | 30 min |
| Training (50 epochs) | 3-4 hours | 30-45 min |
| Evaluation | 10 min | 10 min |
| **Total** | **4 hours** | **1.5 hours** |

---

## 🎓 Learning Path

### Week 1: Foundation
- ✓ Run quick start
- ✓ Understand architecture (see README.md)
- ✓ Test on small dataset (100 images)

### Week 2-3: Training
- ✓ Prepare full dataset
- ✓ Train model (50-100 epochs)
- ✓ Evaluate results
- ✓ Run ablation study

### Week 4-5: Optimization
- ✓ Tune hyperparameters
- ✓ Test robustness
- ✓ Cross-dataset evaluation
- ✓ Document results

### Week 6-8: Deployment
- ✓ Polish Streamlit app
- ✓ Create API
- ✓ Write final documentation
- ✓ Deploy model

---

**You're all set! Start with:**
```bash
streamlit run streamlit_app.py
```

Then upload an image to see predictions in action! 🎬

Questions? Check README.md or IMPLEMENTATION_ROADMAP.md for details.
