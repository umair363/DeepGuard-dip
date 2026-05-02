# DeepGuard

DeepGuard is a dual-branch Convolutional Neural Network (CNN) designed for the detection of facial manipulation and deepfakes. It operates by analyzing images across two distinct domains: the **Spatial Domain** (RGB pixels) and the **Frequency Domain** (Fast Fourier Transform). 

This project was built to address the growing challenge of identifying sophisticated face-swap deepfakes (e.g., FaceForensics++ data) by combining visual artifact detection with frequency spectrum analysis.

## Architecture

The model architecture utilizes a dual-branch approach:
1. **Spatial Branch**: A MobileNetV2-based feature extractor that identifies pixel-level blending artifacts, warping, and texture inconsistencies.
2. **Frequency Branch**: A custom lightweight CNN that processes the 2D FFT magnitude spectrum to identify periodic upsampling artifacts (checkerboard patterns) left behind by generative models.
3. **Fusion Layer**: Concatenates features from both domains and passes them through dense layers for final binary classification.

## Installation

### Prerequisites
- Python 3.8+
- Requirements listed in `requirements.txt`

```bash
git clone https://github.com/umair363/DeepGuard-dip.git
cd DeepGuard-dip
pip install -r requirements.txt
```

## Dataset Preparation

This model is optimized for the **140k Real and Fake Faces** dataset (based on FaceForensics++). 

To download the dataset via the Kaggle API:
```bash
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip -d ./data/
```

## Training

The training pipeline uses a custom, memory-efficient Keras `Sequence` generator to compute FFTs and resize images on-the-fly, allowing for training on massive datasets without OOM errors.

To begin training:
```bash
python train_model.py
```
*Note: Parameters such as batch size, learning rate, and directory paths can be modified in `config.yaml`.*

## Web Interface

DeepGuard includes a Streamlit-based web interface for real-time inference, visualization of the FFT spectrum, and confidence breakdown.

To run the web app locally:
```bash
streamlit run streamlit_app.py
```

## Limitations & Future Work

- **Domain Generalization**: DeepGuard achieves ~98.5% validation accuracy on in-distribution data (FaceSwap techniques). However, detecting fully synthetic AI-generated faces (e.g., StyleGAN, Midjourney) remains an open challenge due to domain shift.
- **Inference Speed**: Current inference runs efficiently on CPU, but further optimization via TFLite quantization is planned for mobile/edge deployment.

## License

This project is intended for research and educational purposes.
