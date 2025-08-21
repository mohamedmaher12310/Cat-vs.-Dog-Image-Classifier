# Cats vs Dogs Classifier 🐱🐶

This project implements an image classifier to distinguish between **cats** and **dogs** using a **pre-trained MobileNetV2 model** from [TensorFlow Hub](https://www.tensorflow.org/hub).  
It leverages **transfer learning** to achieve good accuracy with a small subset of the dataset.  

---

## 📌 Project Overview
- Uses the **Cats vs Dogs dataset** from [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs).
- Pre-trained **MobileNetV2** is used as a **feature extractor**.
- A custom dense layer is trained on top of extracted features.
- Includes visualization of training accuracy and loss.
- Supports single image prediction with confidence score.

---

## 🛠️ Requirements
Install the dependencies before running the project:

```bash
pip install tensorflow tensorflow-hub tensorflow-datasets matplotlib numpy
```
📂 Dataset

Dataset: cats_vs_dogs (via TensorFlow Datasets)

Subset used:

10% for training

5% for validation


🚀 How It Works

Data Loading & Preprocessing

Load dataset with TensorFlow Datasets.

Resize all images to 224x224.

Normalize pixel values to range [0,1].

Model

Pre-trained MobileNetV2 (feature extractor).

Classification head: Dense layer with 2 outputs (cat/dog) using softmax.

Training

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metrics: Accuracy

Trained for 6 epochs.

Evaluation & Visualization

Plots training vs validation accuracy and loss.

Shows predictions on sample validation images.

Prediction

Custom function predict_image(image_path, model)
Loads an external image and predicts whether it's a cat or a dog with confidence score.

📊 Training Results

Achieved reasonable accuracy on a small training subset.

Example training curves are plotted (accuracy & loss).


🖼️ Example Usage
Predict an image

```python
predicted_class, confidence = predict_image('path/to/your/image.jpg', model)
print(f"{predicted_class} ({confidence:.2f}% confidence)")
```
Example Output:

```python
This image most likely belongs to 'dog' with a 92.45% confidence.
```


📌 Next Steps

Train on the full dataset for better accuracy.

Experiment with fine-tuning MobileNetV2 (unfreeze layers).

Try other TensorFlow Hub models (e.g., EfficientNet, ResNet).

Deploy as a simple web app with Flask or Streamlit.

📜 License

This project is released under the MIT License.
Feel free to use and modify it for your own work.



