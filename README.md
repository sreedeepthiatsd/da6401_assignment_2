## DA6401 Assignment-2 
# Task:
Build a comprehensive, multi-stage Visual Perception Pipeline. Deliver a cohesive system capable of detecting, classifying, and segmenting subjects within an image.
Use the Oxford-IIIT Pet Dataset. This is a rich dataset that provides class labels (breed), bounding boxes (head), and pixel-level masks (trimaps). The link for this dataset is given here: https://www.robots.ox.ac.uk/~vgg/data/pets/

* W&B report link: 
* Github report link:

## Multi-Task Visual Perception Pipeline

This project implements a **complete visual perception pipeline** capable of:

- Image classification (pet breed)
- Object localization (bounding box)
- Semantic segmentation (pixel-wise mask)

The system is built using the **Oxford-IIIT Pet Dataset** and follows a **multi-task learning framework with a shared VGG11 backbone**.

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

We use the **Oxford-IIIT Pet Dataset**, which provides rich annotations for image understanding tasks.

### 📁 Dataset Features

- 🐶 **37-class breed labels**  
  Each image is labeled with one of 37 pet breeds (cats and dogs).

- 📦 **Bounding box annotations**  
  Provides object localization by specifying bounding boxes around pets.

- 🎯 **Pixel-level segmentation masks**  
  Fine-grained masks indicating the exact pixels belonging to the pet.

### 🔧 Usage in This Project

- Classification → using breed labels  
- Object Detection → using bounding boxes  
- Segmentation → using pixel-level masks

## Model Architecture

Our model follows a **multi-task learning framework** that performs classification, localization, and segmentation in a single forward pass.

---

### 1️⃣ Shared Encoder: VGG11

- Custom implementation of **VGG11**
- Includes:
  - Batch Normalization
  - ReLU activations
  - MaxPooling layers

**Output Feature Map**:  [B,512,7,7]

---

### 2️⃣ Classification Head

- Fully connected layers:
    *4096 → 4096 → 37
  - Uses custom dropout layer
- Outputs class logits for 37 pet breeds

---

### 3️⃣ Localization Head

- Predicts bounding box coordinates:[x_center, y_center, width, height]
- Uses **sigmoid activation** for normalized outputs

---

### 4️⃣ Segmentation Head (U-Net Style)

- **Encoder:** VGG11 (shared)
- **Decoder:**
  - Transposed Convolutions (Upsampling)
  - Conv + ReLU + Dropout

- Output:
- [B,num_classes,224,224] 
  
---

### 5️⃣ Multi-Task Model

A single forward pass produces outputs for all tasks:

```python
outputs = {
    "classification": logits,
    "localization": bbox,
    "segmentation": mask
}
```
---
## Training details
- Adam Optimiser is used
- Cross Entropy Loss, Mean Square Error, IOU Loss is used in training during various tasks
## Design Choices
- Dropout is applied in fully connected layers and decoder to reduce overfitting
- Batch Normalization is applied after every convolutional layer for better performance
- VGG11 Encoder is shared for all tasks
- In Segmentation task, implemented CrossEntropy with class weighting. It handles class imbalance
- Current training uses reduced dataset for faster execution
