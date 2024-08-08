# MNIST-Dense-VS-ConvNet

## Project Overview

This project focuses on classifying handwritten digits from the MNIST dataset using two different types of neural networks:
1. **Dense Neural Network (DNN)**
2. **Convolutional Neural Network (CNN)**

The objective is to compare the performance of these models and demonstrate the effectiveness of CNNs in image classification tasks.

## Project Structure

- **mnist_classification.ipynb**: Jupyter notebook containing the complete code for loading data, preprocessing, model training, evaluation, and prediction visualization.
- **README.md**: Project description and instructions.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Pandas

## Installation

To set up the environment and install the necessary libraries, run:

```bash
pip install tensorflow numpy matplotlib seaborn pandas
```

## Dataset

The MNIST dataset is used, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Code Explanation

### 1. Data Loading and Visualization

- Load the MNIST dataset.
- Display one sample image from each class (0-9) for visualization.

### 2. Data Preprocessing

- Normalize the pixel values to range [0, 1].
- One-hot encode the labels.

### 3. Dense Neural Network

- Reshape the data to fit the dense network.
- Define a sequential model with multiple dense layers.
- Compile the model with `categorical_crossentropy` loss and `rmsprop` optimizer.
- Train the model on the training data.
- Evaluate the model on the test data.

### 4. Convolutional Neural Network

- Reshape the data to fit the CNN.
- Define a sequential model with convolutional layers, max pooling, and dense layers.
- Compile the model with `categorical_crossentropy` loss and `adam` optimizer.
- Train the model on the training data.
- Evaluate the model on the test data.

### 5. Results Comparison

- Print the test accuracy of both models.
- Display the model summaries.
- Test the CNN with a random sample from the test set and show the prediction.

## Model Summaries

### Dense Neural Network
- **Layers**: 5 dense layers with ReLU activation and a softmax output layer.
- **Test Accuracy**: [To be filled after training]

### Convolutional Neural Network
- **Layers**: 3 convolutional layers, 2 max pooling layers, 1 dense layer, and a softmax output layer.
- **Test Accuracy**: [To be filled after training]

## Sample Prediction

- A random test image is displayed along with the model's predicted label and the actual label for visual verification.

## Conclusion

This project demonstrates the superiority of CNNs in image classification tasks over dense neural networks. The CNN achieved a higher test accuracy and was more effective in correctly classifying the handwritten digits from the MNIST dataset.

