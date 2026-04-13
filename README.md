## DA6401 Assignment-2 
# Task
Build a comprehensive, multi-stage Visual Perception Pipeline. Deliver a cohesive system capable of detecting, classifying, and segmenting subjects within an image.
Use the Oxford-IIIT Pet Dataset. This is a rich dataset that provides class labels (breed), bounding boxes (head), and pixel-level masks (trimaps). The link for this dataset is given here: https://www.robots.ox.ac.uk/~vgg/data/pets/



* W&B report link:
* Github link:

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

