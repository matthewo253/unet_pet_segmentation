# Oxford-IIIT Pet Semantic Segmentation Pipeline

A robust PyTorch-based semantic segmentation pipeline built for the **Oxford-IIIT Pet dataset**. The project utilizes a **U-Net** architecture with a pretrained ResNet-34 encoder, a combined Cross-Entropy and Dice loss function, spatial data augmentations, and multi-class Run-Length Encoding (RLE) for Kaggle competition submissions.

---

## 📁 Repository Structure

* **`data.py`**: Handles dataset downloading via `torchvision`, custom segmentation dataset wrapping, resizing, label normalization into classes `[0, 1, 2]`, and data loader setup.
* **`main.py`**: Contains the model training script utilizing `segmentation_models_pytorch` (U-Net with ResNet34), custom weighted Cross-Entropy + Dice loss, augmentation loops (horizontal flips, random rotations), Cosine Annealing learning rate scheduling, and mIoU validation tracking.
* **`inference.py`**: Loads the best saved model checkpoint (`best_model.pth`), performs batch-free inference on test images, upscales predictions to original image resolution via bilinear interpolation, and exports per-image numpy masks.

---

## 🚀 Key Features

* **Dataset & Preprocessing**: Downloads the Oxford-IIIT Pet dataset automatically, standardizes image sizes to $224 \times 224$, and safely clips/maps class masks.
* **Advanced Loss Function**: Implements a hybrid loss combining weighted Cross-Entropy and class-weighted Dice loss to handle class imbalance.
* **Data Augmentation**: Incorporates on-the-fly random horizontal flips and rotations during training to improve generalization.
* **Evaluation Metric**: Computes multi-class **mean Intersection over Union (mIoU)** across validation epochs.

---

## 🛠️ Requirements & Installation

Make sure you have PyTorch and the required libraries installed:

```bash
pip install torch torchvision segmentation-models-pytorch numpy pillow
