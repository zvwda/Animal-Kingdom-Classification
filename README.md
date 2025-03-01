# Animal Classification Project

## Description  
This project trains deep learning models to classify animals based on image features. Using a dataset of animal images, the model predicts the correct class among 18 categories. The training process leverages data augmentation and model fine-tuning to enhance accuracy.  

---

## Dataset  
Access the dataset for training and testing on Kaggle: [Dataset Link](https://www.kaggle.com/competitions/animal-kingdom-classification/overview)  

- **Training Set**: [Download Here](https://drive.google.com/file/d/1Z8H7gYduXvLFmomW540OuWyJEuonwLGu/view?usp=sharing)  
- **Testing Set**: [Download Here](https://drive.google.com/file/d/1_3uRs3dQmQlyVKOBiEyYiAWLxh9P8If3/view?usp=sharing)  

---

## Model Overview  

### Vision Transformer (ViT)  
- Images resized to **224×224**  
- **Data Augmentations**:  
  - Random flips  
  - Color jittering  
  - Affine transformations  
- **Architecture**:  
  - Patch Embedding  
  - Multi-Head Attention  
  - MLP Layers  
  - Custom classification head for 18 classes  
- **Training Details**:  
  - Optimizer: **Adam**  
  - Learning Rate: **1×10⁻³ (head), 1×10⁻⁵ (other layers)**  
  - Epochs: **5**  
  - Batch Size: **4**  
  - Best Validation Accuracy: **99.86%**  

### ResNet Model  
- **Pretrained ResNet50** used for feature extraction, but the classification head was replaced and trained from scratch.  
- **Training Details**:  
  - Optimizer: **Adam**  
  - Learning Rate: **1×10⁻⁵**  
  - Batch Size: **4**  
  - Epochs: **10**  
  - Best Validation Accuracy: **97.47%**  

---

## Key Results  
- **Vision Transformer:** **99.86% validation accuracy**  
- **ResNet Model:** **97.47% validation accuracy**  

Confusion matrix and loss curves included to visualize performance.  

---

## Challenges & Solutions  
- **Class Imbalance** → Handled using augmentation techniques  
- **Overfitting** → Reduced using dropout layers and extensive data augmentation  
- **Limited Resources** → Adjusted batch size to optimize training  

---

## Conclusion  
This project successfully classified animal species using **Vision Transformer** and **ResNet** models. **Only pretrained weights were used for initialization**, while all models were fine-tuned with custom layers. This approach ensured maximum accuracy and adaptability.  

