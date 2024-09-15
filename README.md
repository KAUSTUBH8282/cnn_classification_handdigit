# CNN Model for MNIST Handwritten Digit Classification

This project implements a Convolutional Neural Network (CNN) in PyTorch for classifying handwritten digits from the MNIST dataset. The model is trained on the MNIST dataset using a two-layer convolutional architecture and achieves classification accuracy on the test set.

## Requirements

The following libraries and frameworks are required to run the project:

- Python 3.x
- PyTorch
- torchvision
- torch
- numpy

You can run this code on your local machine or directly on **Google Colab**. Running it on Google Colab provides a free GPU for training, which speeds up the process.

## Running on Google Colab

To run this project on Google Colab:

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and copy the contents of the Python script into the cells.
3. Ensure that the runtime type is set to **GPU** for faster training:
   - Go to `Runtime` -> `Change runtime type` -> Select **GPU** from the dropdown.
4. Install the required libraries (if not already installed):

   ```bash
   !pip install torch torchvision
Proceed with the training and testing steps as described below.
Model Architecture
The CNN model consists of the following layers:

conv1: 2D convolution layer with 32 output channels and 3x3 kernel, input size 1x28x28.
conv2: 2D convolution layer with 64 output channels and 3x3 kernel.
Max Pooling: 2x2 pooling layer.
Fully connected layer 1: 64 * 7 * 7 input features, 128 output features.
Fully connected layer 2: 128 input features, 10 output features (for the 10 digit classes).
Activation: ReLU for non-linear activation between layers and Log Softmax for output classification.
Dataset
The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9). The dataset is automatically downloaded using torchvision.datasets.MNIST with the following transforms:

ToTensor(): Converts the image to a PyTorch tensor.
Normalize(): Normalizes the image using the mean and standard deviation of the dataset.

Training and Testing
The training and testing processes are handled by the train() and test() functions respectively.

Training
The model is trained for a specified number of epochs (default: 10) with the Adam optimizer and Cross Entropy Loss. The loss and training progress are printed every 100 batches.

Testing
The model is evaluated on the test dataset after each epoch, and the average loss and accuracy are displayed.

If you are running on Google Colab, copy the script contents into the Colab notebook and execute the cells. Make sure to enable GPU for faster training.

Results
At the end of training, the model's performance is evaluated on the test set, displaying metrics like average loss and accuracy.
