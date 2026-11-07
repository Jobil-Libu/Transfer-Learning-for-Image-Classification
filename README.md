# CIFAR-10 Image Classification using Transfer Learning

An image classification project that applies **Transfer Learning with a pretrained ResNet50 CNN** to classify images from the CIFAR-10 dataset into 10 distinct categories.

The project uses ImageNet-pretrained weights to extract visual features and a custom classification head consisting of Global Average Pooling, Dense layers, Batch Normalization, ReLU activation, and Dropout.

It implements two training phases:
- **Feature Extraction:** Freezing the pretrained ResNet50 backbone and training the custom classification head.
- **Fine-Tuning:** Unfreezing selected deeper layers of ResNet50 and retraining them with a lower learning rate to adapt the model to CIFAR-10.

### Key Features
- CIFAR-10 dataset preparation and visualization
- Image resizing to 224 × 224
- ImageNet normalization and data augmentation
- Pretrained ResNet50 architecture
- He weight initialization
- L2 regularization and Dropout
- AdamW optimizer
- ReduceLROnPlateau learning rate scheduling
- Early Stopping and Model Checkpointing
- GPU-based model training
- Performance comparison between feature extraction and fine-tuning
- Confusion matrix, classification report, and per-class accuracy evaluation

### Technologies Used
Python | TensorFlow | Keras | NumPy | Matplotlib | Scikit-learn | Kaggle Notebook

### Objective
To explore how transfer learning can leverage pretrained CNN architectures for image classification and understand the impact of fine-tuning, regularization, and training strategies on model generalization.
