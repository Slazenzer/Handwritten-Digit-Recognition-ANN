# Handwritten Digit Recognition using ANN

## Introduction
This project implements a Handwritten Digit Recognition system using an Artificial Neural Network (ANN) with TensorFlow and Keras. It trains on the MNIST dataset and allows users to predict handwritten digits from custom images.

## Dataset
The project uses the **MNIST dataset**, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size 28x28 pixels.

## Technologies Used
- **Python**: Programming language
- **TensorFlow/Keras**: Machine learning framework for neural networks
- **NumPy**: For numerical operations
- **Matplotlib**: For data visualization
- **OpenCV (cv2)**: For image processing
- **OS Module**: For file handling

## Libraries Used
### 1) os
   - Used for file handling, checking file paths, and managing external digit images.

### 2) cv2 (OpenCV)
   - Used for reading, processing, and inverting external images for digit recognition.

### 3) numpy
   - Provides support for large arrays and matrices, used for preprocessing and handling image data.

### 4) matplotlib.pyplot
   - Used for displaying images and visualizing results.

### 5) tensorflow
   - Used for creating and training the Artificial Neural Network (ANN) model.

## Model Architecture
The ANN model consists of:
- **Input Layer**: Flatten layer to convert 28x28 images into a single 784-dimensional vector.
- **Hidden Layers**: Three dense layers, each with 128 neurons and ReLU activation.
- **Output Layer**: A softmax layer with 10 neurons (for digits 0-9).

## Steps to Run the Project
1. **Install dependencies**:
2. **Run the script**:
3. The model will train on the MNIST dataset for 5 epochs.
4. After training, the model can predict digits from custom images in the `Digits/` folder.

## Predicting Custom Images
- Place images of handwritten digits (28x28 grayscale PNG format) inside the `Digits/` directory.
- The script will load and predict the digit from each image.
- Predictions are displayed using Matplotlib.

## Future Improvements
- Convert the model to use **CNN** for better accuracy.
- Add GUI for easier digit input.
- Deploy as a web app.

## License
This project is open-source and free to use.