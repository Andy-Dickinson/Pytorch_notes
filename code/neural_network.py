import os  # handling file and directory paths
import torch  # tensor operations, basic operations (core PyTorch utilities), autograd functionality
from torch import nn  # classes and functions for building neural net layers and modules
from torch.utils.data import DataLoader  # load data in batches, shuffle, handle multiprocessing
from torchvision import (
    datasets,  # preloaded datasets (e.g., CIFAR-10, MNIST)
    transforms  # transforms data, usually image preprocessing and augmentation (e.g. images to tensors)
)

device = (
    # cuda is available on systems with NIVIDIA GPUs
    "cuda" if torch.cuda.is_available()
    # mps is available on macOS systems that support Metal (Apple silicon e.g. M1/M2 chips)
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # converts each image to a 1D vector e.g. [1,28,28] -> [1, 784]
        self.linear_relu_stack = nn.Sequential(  # container module to stack layers in sequence. Each output becomes input to next
            nn.Linear(28 * 28, 512),  # each of the 784 (28*28) input features is connected to each of the 512 output features
            nn.ReLU(),  # element-wise activation function, introducing non-linearity into the model. Shape of tensor remains the same, but values are transformed
            nn.Linear(512, 512),  # transforms 512-dimensional input to another 512-dimensional output to learn new representation of the features
            nn.ReLU(),
            nn.Linear(512, 10),  # maps 512-dimensional input to 10-dimensional output corresponding to logits (in forward()) for each of the 10 classes in the classification task
        )

    # do not directly call model.forward()
    # it is automatically called (via nn.Module) when we pass the model input data
    def forward(self, x):
        x = self.flatten(x)
        logits: object = self.linear_relu_stack(x)
        return logits


# create instance and move to available device
# do this before training or using the model
model = NeuralNetwork().to(device)

X = torch.rand(1, 28, 28, device=device)  # [batch size, height, width], [1 image, 28 x 28]
x_logits = model(X)  # internally calls forward() and returns the networks logits (raw output values) as tensor

print(f"Input X: \n {X}")
print(f"Shape input X: {X.shape} \n")

print(f"logits (model raw outputs): \n {x_logits}")
print(f"Shape logits: {x_logits.shape} -> [batch_size, raw_predicted_values_for_each_of_the_10_classes]\n")

"""
Softmax normalizes the logits to a probability distribution over the classes, applied across the class dimension (dim=1)
pred_probab is a tensor the same shape as logits [1,10], 
but now represents probabilities instead of raw values. 
Each value is between 0 and 1 and will sum to 1 along class dimension """
pred_probab = nn.Softmax(dim=1)(x_logits)
print(f"pred_probab: \n {pred_probab}")
print(f"pred_probab shape: \n {pred_probab.shape} \n")

y_pred = pred_probab.argmax(1)  # returns index of the highest probability in dim=1 as tensor
print(f"Predicted class index: {y_pred.item()}")
