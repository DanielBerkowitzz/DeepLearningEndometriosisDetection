# Deep Learning for Non-Invasive Endometriosis Detection

This repository presents a deep learning model developed for the non invasive diagnosis of endometriosis based on axial pelvic MRI scans. The project leverages transfer learning with DenseNet-121, a convolutional neural network pretrained on ImageNet, which was fine tuned to classify axial T1 weighted and T2 weighted MRI slices as either positive or negative for endometriosis. Given the significant variability in lesion appearance, the limited size of labeled medical datasets, and the subtle imaging features of superficial lesions, the project focuses on data augmentation and modality specific training. By integrating multiple MRI contrasts and selectively unfreezing the deeper layers of the model, this approach achieves high classification performance and demonstrates that deep learning can support radiologists in identifying endometriotic patterns, potentially reducing diagnostic delays and the need for invasive procedures such as laparoscopy.

---

## Overview

- **Model**: DenseNet-121 with transfer learning and fine-tuning
- **Input**: Axial MRI slices (T1, T2, and combined)
- **Output**: Binary classification (Endometriosis / No Endometriosis)
- **Dataset**: UT-EndoMRI (Zenodo), 130 labeled patient cases
- **Environment**: Google Colab
- **Results**: Achieved 0.9346 accuracy and 0.3198 loss on the combined modalities

---

## Dataset

This project uses the **UT-EndoMRI dataset**, a publicly available collection of labeled pelvic MRI scans, including:

- T1-weighted, T2-weighted, T1FS, and T2FS sequences
- Binary class labels: endometriosis / no label of endometriosis

Citation:
Liang, X., Alpuing Radilla, L. A., Khalaj, K., Mokashi, C., Guan, X., Roberts, K. E., Sheth, S. A., Tammisetti, V. S., & Giancardo, L. (2025). UTHealth - Endometriosis MRI Dataset (UT-EndoMRI) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15750762

> Due to dataset licensing and privacy constraints, MRI images are **not distributed** via this repository. The code assumes the data is available in a mounted Google Drive folder.

---

## Model Architecture

- **Base model**: DenseNet-121, pretrained on ImageNet
- **Modifications**:
  - Final classifier replaced for binary output
  - `denseblock4` and `norm5` unfrozen for medical specific fine tuning
- **Loss**: CrossEntropy with Label Smoothing (ε = 0.1)
- **Regularization**:
  - Dropout (0.5)
  - Batch Normalization
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Based on validation loss with patience limit of 5

---

## Data Augmentation

To mitigate the small dataset size and improve generalization:

- Slice based sampling around central pelvic region
- Rotation of 90, 180, 270 degrees for each slice
- Horizontal flips and 15 degrees
- Brightness and contrast jitter

Resulting in about 30,000 augmented samples across training/validation/test splits.

---

## Training Results

Evaluation was conducted separately on T1, T2, and combined modalities:

| Configuration          | Accuracy | Precision | Recall | F1 Score |
| T1-only                | 93.39%   | 95.52%    | 90.77% | 93.08%   |
| T2-only                | 92.47%   | 93.53%    | 91.42% | 92.46%   |
| T1+T2 combined         | **93.46%** | **94.13%** | **92.03%** | **93.07%** |

All results were obtained with batch size 64 and 75–100 training epochs.

---

## Usage Instructions

### Requirements

- Python 3.10, PyTorch, torchvision, numpy, matplotlib

### Execution with Google Colab

1. Open `model_t1_t2.ipynb` in Colab
2. Mount your Google Drive (`drive.mount()` cell)
3. Ensure the following folder structure in Drive:
/MyDrive/
├── dataset_t1_new/
└── dataset_t2_new/
5. Run all cells in order. The notebook includes:
- Data loading & augmentation
- Model definition
- Training & validation loop
- Test evaluation
- Confusion matrix & metrics
- Prediction on new MRI slice

---

## Research Contribution

This project was completed as part of a senior capstone at Braude College of Engineering, exploring machine learning applications in medical imaging. The findings demonstrate that convolutional neural networks, when combined with appropriate preprocessing and transfer learning, can provide a promising results for non invasive endometriosis diagnosis.

---

## Authors

- **Daniel Berkowitz**
- **Tal Turjeman**

Supervised by: Prof. Miri Weiss Cohen  
Capstone Project Phase B | Braude College | 2025

---

## License

This project is provided for academic and research purposes only. Redistribution or clinical application requires permission from the authors.
