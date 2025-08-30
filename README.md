<<<<<<< HEAD
# Deepfake Detection System

A comprehensive deepfake detection system using PyTorch and computer vision techniques. This project includes both heuristic-based detection and machine learning models trained on real vs fake image datasets.

## Features

- **Heuristic Detection**: Uses frequency analysis, color correlation, and compression artifacts
- **ML Model**: ResNet18-based classifier trained on real/fake image pairs
- **Web Interface**: Streamlit app for easy image/video upload and analysis
- **Face Detection**: OpenCV-based face cropping for focused analysis
- **Multiple Interfaces**: Streamlit web app, HTML interface, and CLI tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/xavierstephy/deep_fake_detector.git
cd deep_fake_detector

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Web Interface (Recommended)
```bash
streamlit run app.py
```

#### Command Line
```bash
# Test a single image
python app.py path/to/image.jpg

# Test the trained model
python test_model.py
```

#### Training Your Own Model
```bash
# Organize your dataset (if needed)
python organize_dataset.py

# Train the model
python train_model.py --data-dir dataset2 --device cpu --model resnet18 --epochs 10
```

## Project Structure

```
deep_fake_detector/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── test_model.py          # Model testing script
├── organize_dataset.py    # Dataset organization utility
├── requirements.txt       # Python dependencies
├── index.html            # HTML interface
├── model.pth             # Trained model (after training)
└── dataset2/             # Training dataset
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
```

## Dataset Requirements

The training script expects a dataset with the following structure:
```
dataset/
├── train/
│   ├── real/     # Real images
│   └── fake/     # Fake/deepfake images
└── val/
    ├── real/     # Validation real images
    └── fake/     # Validation fake images
```

If you only have a `train` folder, the script will automatically create a validation split.

## Model Architecture

- **Base Model**: ResNet18 with ImageNet pretrained weights
- **Output**: 2-class classification (Real vs Fake)
- **Input**: 224x224 RGB images
- **Augmentation**: Random horizontal flip, color jittering
- **Optimizer**: AdamW with cosine annealing scheduler

## Detection Methods

### 1. Heuristic Analysis
- **Laplacian Variance**: Detects blur/compression artifacts
- **High-Frequency Energy**: Identifies unnatural frequency patterns
- **Color Channel Correlation**: Detects abnormal color relationships
- **DCT Consistency**: Analyzes compression artifacts

### 2. Machine Learning
- **Feature Extraction**: Automatic feature learning from training data
- **Classification**: Binary classification with confidence scores
- **Face Cropping**: Optional face detection for focused analysis

## Performance

The trained model typically achieves:
- **Accuracy**: 60-80% on validation set
- **AUC**: 0.7-0.9 depending on dataset quality
- **Inference Time**: ~100ms per image on CPU

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with `streamlit run app.py`

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- PyTorch and torchvision for the deep learning framework
- OpenCV for computer vision utilities
- Streamlit for the web interface
- ResNet architecture from torchvision models

## Contact

For questions or issues, please open an issue on GitHub.
=======
# deep_fake_detector
>>>>>>> 0c57338d10119562221af77a603da511f81f8170
