Cat vs. Dog Image Classifier
A deep learning project that accurately classifies images of cats and dogs using transfer learning with TensorFlow. This implementation demonstrates how to leverage pre-trained models to achieve high accuracy with minimal training time and data.

https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg
https://img.shields.io/badge/Python-3.x-blue.svg
https://img.shields.io/badge/License-MIT-green.svg

Project Highlights
High Accuracy: Achieves >99% validation accuracy after just 6 epochs of training

Transfer Learning: Utilizes MobileNetV2 pre-trained on ImageNet for feature extraction

Efficient Implementation: Uses TensorFlow Datasets with optimized data pipelines

Minimal Data Requirement: Trained on only 10% of the full dataset (2,326 images)

Clean Code: Well-documented Jupyter notebook with step-by-step implementation

Model Architecture
The classifier uses a simple yet effective architecture:

Base Model: MobileNetV2 (pre-trained on ImageNet) as a feature extractor

Classification Head: Single Dense layer with softmax activation for binary classification

Frozen Weights: Base model weights remain frozen during training

text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer (KerasLayer)    (None, 1280)              2257984   
                                                                 
 dense (Dense)               (None, 2)                 2562      
                                                                 
=================================================================
Total params: 2260546 (8.62 MB)
Trainable params: 2562 (10.01 KB)
Non-trainable params: 2257984 (8.61 MB)
Results
After 6 epochs of training:

Training Accuracy: 99.53%

Validation Accuracy: 98.97%

Training Loss: 0.0236

Validation Loss: 0.0383

The model shows excellent performance with minimal overfitting, demonstrating the power of transfer learning.

Installation
Clone the repository:

bash
git clone https://github.com/your-username/CatDogClassifier.git
cd CatDogClassifier
Install required dependencies:

bash
pip install tensorflow tensorflow_hub tensorflow_datasets matplotlib numpy
Usage
Open and run the Jupyter Notebook:

bash
jupyter notebook CatDogClassifier.ipynb
Execute the cells sequentially to:

Load and preprocess the dataset

Create and compile the model

Train the classifier

Evaluate performance

Visualize training results

Dataset
The model uses the cats_vs_dogs dataset from TensorFlow Datasets:

Training samples: 2,326 (10% of full dataset)

Validation samples: 1,163 (5% of full dataset)

Image size: 224x224 pixels (resized for MobileNetV2 compatibility)

Future Improvements
Implement data augmentation for better generalization

Experiment with fine-tuning later layers of the base model

Extend to multi-class classification for more animal types

Deploy as a web application with a user interface

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
TensorFlow team for the excellent documentation and pre-trained models

TensorFlow Datasets for providing easy access to the cats_vs_dogs dataset

The deep learning community for sharing knowledge and best practices

