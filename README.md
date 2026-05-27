# CIFAR-10 Image Compression and Classification using Autoencoders & CNN

## Project Overview

This project demonstrates the use of **Convolutional Autoencoders** for image compression and investigates how compressed image representations affect classification performance on the **CIFAR-10 dataset**.

The workflow consists of:

- Training a **Convolutional Autoencoder** to compress CIFAR-10 images
- Performing **hyperparameter tuning** to find the optimal compression architecture
- Extracting the encoder as a compressed feature generator
- Training a **baseline CNN classifier** on original CIFAR-10 images
- Training another CNN classifier on compressed image representations
- Comparing the performance of both classification models

This project explores the trade-off between **dimensionality reduction** and **classification accuracy**.

---

## Problem Statement

To compress data using autoencoders in Keras, the model is trained such that the input and output are identical.

This project aims to:

1. Compress CIFAR-10 images using a convolutional autoencoder
2. Determine the optimal hyperparameters for the compression model
3. Use the compressed representations for image classification
4. Compare classification performance against a baseline CNN trained on original images

---

## Dataset

### CIFAR-10

The CIFAR-10 dataset contains **60,000 color images** belonging to **10 classes**.

- Training images: 50,000
- Test images: 10,000
- Image size: 32 × 32 × 3 (RGB)

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Architecture

### Autoencoder Pipeline

```text
Input Image (32x32x3)
        ↓
Convolution Layer
        ↓
Max Pooling
        ↓
Convolution Layer
        ↓
Max Pooling
        ↓
Compressed Representation
        ↓
Convolution Layer
        ↓
UpSampling
        ↓
Convolution Layer
        ↓
UpSampling
        ↓
Reconstructed Image
```

### Baseline CNN

```text
Original CIFAR-10 Images
        ↓
Conv2D
        ↓
MaxPooling
        ↓
Conv2D
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense
        ↓
Softmax Output
```

### Compressed CNN

```text
Compressed Encoder Features
        ↓
Conv2D
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense
        ↓
Softmax Output
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
- CIFAR-10 Dataset

---

## Hyperparameter Tuning

The following hyperparameters were explored:

- Filter size
- Pool size
- Number of convolution filters
- Number of encoder/decoder layers
- Number of epochs

Example configurations:

- Filter Sizes: 3, 5
- Pool Size: 2
- Filter Configurations: [16,32], [32,64], [32,64,128]
- Epochs: 10+

---

## Results

| Model          | Accuracy |
|----------------|----------|
| Baseline CNN   | ~64%     |
| Compressed CNN | ~38%     |

---

## Analysis

### Observations

- The baseline CNN significantly outperformed the compressed CNN.
- The autoencoder successfully reconstructed images but lost important class-discriminative information.
- CIFAR-10 images are already small (32×32), so aggressive compression causes substantial information loss.

### Conclusion

Autoencoders are effective for image compression and representation learning, but compressed latent representations are not always optimal for downstream classification without careful architecture tuning.

---

## Installation

```bash
git clone https://github.com/VivekSingh1201/autoencoder-classification-and-compression-with-baseline-classier.git
cd cifar10-autoencoder-classification
```

Create environment:

```bash
python -m venv venv
```

Activate:

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

**Notebook:**
```bash
jupyter notebook
```

**Script:**
```bash
python main.py
```

---

## Future Improvements

- Deeper autoencoders
- Denoising autoencoders
- Variational autoencoders (VAE)
- Better compressed classifier architectures
- Transfer learning
- Learning rate tuning
- Batch normalization experiments

---

## Author

**Vivek Kumar Singh**

---

## License

Educational and research use.
