# Image Classification with LeNet-5
![image](https://github.com/vaishnav-mk/lenet_mnist/assets/84540554/880c0268-4bb0-474f-8299-7aaa6887b53a)


This repository contains code for a simple image classification model using an enhanced LeNet-5 architecture. LeNet-5 is a convolutional neural network (CNN) designed by Yann LeCun and his collaborators for handwritten digit recognition. The enhanced version incorporates modern practices like ReLU activation, Batch Normalization, and Dropout for improved performance.

## Enhanced LeNet-5 Architecture

The enhanced LeNet-5 architecture follows the same structure as the original but incorporates the following modifications:

### Feature Extractor

1. **Convolutional Layer 1**: Input images (32x32 for LeNet-5) are convolved with six kernels of size 5x5, followed by a Rectified Linear Unit (ReLU) activation function. This is then subsampled using average pooling.

2. **Convolutional Layer 2**: The output from the first layer is convolved with sixteen kernels of size 5x5, followed by ReLU activation and subsampling.

### Batch Normalization

Batch Normalization is applied after each convolutional layer to normalize the input, helping with training stability and accelerating convergence.

### Classifier

1. **Flatten Layer**: The output from the second subsampling layer is flattened into a vector.

2. **Fully Connected Layers with ReLU Activation**: Three fully connected layers with ReLU activations. The first two layers have 120 and 84 neurons, respectively, and the last layer has 10 neurons corresponding to the 10 classes in the MNIST dataset.

### Dropout

Dropout is applied to the fully connected layers during training to prevent overfitting. It randomly sets a fraction of input units to zero at each update, forcing the network to learn more robust features.

### Data Transforms

Data transforms play a crucial role in preparing the input data for the model. The following transforms were applied:

- **Random Rotation**: Images in the training set are randomly rotated between -15 and 15 degrees to increase variety in the training data.

- **Random Inversion**: With a probability of 50%, colors are randomly inverted to further diversify the dataset.

- **Random Grayscale**: Also with a 50% probability, images are randomly converted to grayscale, introducing more variability.

- **ToTensor()**: Converts the images to PyTorch tensors, making them compatible with the model.

The training set utilizes these transforms to augment the dataset, while the validation and test sets only apply the standard `ToTensor()` transform.
