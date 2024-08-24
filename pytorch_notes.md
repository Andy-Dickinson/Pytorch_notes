# <center>Pytorch Notes</center>

---  

### <center>Table of Contents</center>  
|Item|Heading|Subcontents|
|:---:|:---:|:---:|
| **1.** | [Reference Links](#reference-links) ||
| **2.** | [Tensors](#tensors) | [Initialising](#initialising-a-tensor),<br>[Attributes](#attributes-of-a-tensor),<br>[Operations](#operations-on-a-tensor) - indexing, joining, arithmetic etc., |
| **3.** | [Datasets & DataLoaders](#datasets--dataloaders) | [Loading datasets](#loading-datasets),<br>[Transforms](#transforms),<br>[Creating a Custom Dataset](#creating-a-custom-dataset),<br>[Iterating & Visualising the Dataset](#iterating--visualising-the-dataset),<br>[Preparing Data for Training with DataLoaders](#preparing-data-for-training-with-dataloaders) |
| **4.** | [Building a Neural Network](#building-a-neural-network) | [Get Device for Training](#get-device-for-training),<br>[Define the Class](#define-the-class),<br>[Using a Model](#using-a-model) |
| **5.** | [`torch.nn` Module](#torchnn-module) | Basic building blocks for graphs including neural net layers, activation functions and loss functions |
| **6.** | [Activation Functions](#activation-functions) ||
| **7.** | [Automatic Differentiation With Autograd](#automatic-differentiation-with-autograd) | [Compute Gradients](#compute-gradients),<br>[Operations & Tracking](#operations--tracking) |
| **8.** | [Optimising Model Parameters - Train/Test](#optimising-model-parameters---traintest) | [Hyperparameters](#hyperparameters),<br>[Initialise Loss Function](#initialise-loss-function),<br>[Initialise Optimiser](#initialise-optimiser) |
| **9.** | [Loss Functions](#loss-functions) ||
| **10.** | [Optimisers](#optimisers) ||

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Reference Links</u>  

[Pytorch Documentation](https://pytorch.org/docs/stable/index.html)  
[Tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)  

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Tensors</u>

* Similar to **arrays** and **matrices**.  
* Can run on GPUs or other hardware accelerators.  

```python
import torch
import numpy as np
```  

##### <u>Initialising a Tensor</u>  

|Item|Subheading|
|:---:|:---:|
| **1.** | [Directly from data](#directly-from-data) |
| **2.** | [From a NumPy array](#from-a-numpy-array) |
| **3.** | [From another tensor](#from-another-tensor) |
| **4.** | [With random or constant values](#with-random-or-constant-values) |


###### Directly from data:  
```py
# Data type is automatically inferred
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

###### From a NumPy array:  
> Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.  
```py
np_array = np.array(data)
t = torch.from_numpy(np_array)

# NumPy arrays can also be created from tensors
t2 = torch.ones(5)
np_array2 = t2.numpy()
```  

###### From another tensor:  
```py
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden
x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# rand_like by default expects input to be floating-point tensor, but can override as shown here if input does not match
```

###### With random or constant values:  
```py
# Shape is a tuple of tensor dimensions
shape = (2,3,)

rand_tensor = torch.rand(shape) # random floats 0-1 (inclusive 0, exclusive 1)
ones_tensor = torch.ones(shape) # tensor of floats 1.
zeros_tensor = torch.zeros(shape) # tensor of floats 0.
```  

<br>  

##### <u>Attributes of a Tensor</u>  

```py
tensor = torch.rand(3,4)

shape = tensor.shape
datatype = tensor.dtype
device = tensor.device  # device tensor is stored on (CPU, GPU)
```

<br>

##### <u>Operations on a Tensor</u>  

[Available operations](https://pytorch.org/docs/stable/torch.html) includes **arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling** (generating random values, seeding, sample from various distributions, random selection, noise addition, shuffling etc.) and more.  

|Item|Subheading|
|:---:|:---:|
| **1.** | [CPU / GPU device](#cpu--gpu-device) |
| **2.** | [Indexing and slicing](#indexing-and-slicing) |
| **3.** | [Joining tensors](#joining-tensors) |
| **4.** | [Matrix multiplication](#matrix-multiplication) |
| **5.** | [Element-wise product](#element-wise-product) |
| **6.** | [Convert single-element tensors to Python numerical value](#convert-single-element-tensors-to-python-numerical-value) |
| **7.** | [In-place operations](#in-place-operations) |

###### CPU / GPU device:
* By default, operations are run on CPU. Can be run on GPU (typically faster) but need to **explicitly move tensors to the GPU** using `.to` method (after checking for GPU availability).  
* If using Colab, allocate a GPU by going to `Runtime > Change runtime type > GPU`.  
```py
# Move tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Alternatively define when creating tensor
X = torch.rand(1, 28, 28, device=device)  # see building-a-neural-network > get-device-for-training below
```  

###### Indexing and slicing:  
```py
tensor = torch.tensor([[1, 2, 3], 
                       [4, 5, 6]])

First row = tensor[0] 
First column = tensor[:, 0]  # careful how use, see below
Last column = tensor[..., -1] 

tensor[:,1] = 0  # set all 2nd column to zeros

# note difference between [..., ] and [:, ]
3d_tensor = torch.tensor([[[1, 2, 3],
                           [4, 5, 6]],

                          [[7, 8, 9],
                           [10, 11, 12]]])
last_col = 3d_tensor[..., -1]  # Slice last ele of last dimension across all dimensions: tensor([[ 3,  6], [ 9, 12]])
last_ele_2nd_dim = 3d_tensor[:, -1]  # Slice last ele of 2nd dimension for all rows: tensor([[ 4,  5,  6], [10, 11, 12]])
```

###### Joining tensors:  
```py
# torch.cat concatenates a sequence of tensors along a given dimension 
#   - tensors must have the same shape except along the dimension specified by dim
# torch.stack concatenates a sequence of tensors along a new dimension
#   - tensors must have exactly the same shape
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])  # shape: [2, 3]


# torch.cat(dim=0) - vertical concatenation (row-wise)
joined_0 = torch.cat([tensor, tensor, tensor])  # default dim=0
""" tensor([[1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [4, 5, 6]])  shape: [6, 3] """

# torch.cat(dim=1) - horizontal concatenation (column-wise)
joined_1 = torch.cat([tensor, tensor, tensor], dim=1)  
""" tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3],
            [4, 5, 6, 4, 5, 6, 4, 5, 6]])  shape: [2, 9] """

# torch.stack(dim=0) - creates a new dimension at the start (like adding layers)
stacked_0 = torch.stack([tensor, tensor, tensor])  # default dim=0
""" tensor([[[1, 2, 3],
             [4, 5, 6]],

            [[1, 2, 3],
             [4, 5, 6]],

            [[1, 2, 3],
             [4, 5, 6]]])  shape: [3, 2, 3] """  

# torch.stack(dim=1) - creates a new dimension in the middle
stacked_1 = torch.stack([tensor, tensor, tensor], dim=1)
""" tensor([[[1, 2, 3],
             [1, 2, 3],
             [1, 2, 3]],

            [[4, 5, 6],
             [4, 5, 6],
             [4, 5, 6]]])  shape: [2, 3, 3] """
```

###### Matrix multiplication:  
```py
# @ is matrix multiplication operator
y1 = tensor @ tensor.T  # matrix multiplication between tensor and its transpose
y2 = tensor.matmul(tensor.T)  # equivalent to previous operation

# another way to perform matrix multiplication
y3 = torch.rand_like(y1)  # initialises y3 to same shape and data type as y1, but with random values
torch.matmul(tensor, tensor.T, out=y3)  # performs matrix multiplication (same as above), then stores result in y3
```

###### Element-wise product:  
```py
z1 = tensor * tensor  # element-wise multiplication
z2 = tensor.mul(tensor)  # equivalent to previous operation

# another way to perform element-wise multiplication
z3 = torch.rand_like(tensor)  # initialises z3 to same shape and data type as 'tensor' but with random values
torch.mul(tensor, tensor, out=z3)  # performs element-wise multiplication (same as above), then stores result in z3
```

###### Convert single-element tensors to Python numerical value:  
```py
agg = tensor.sum()  # example of aggregating all values of a tensor into one value
agg_item = agg.item()  # converts to float using .item()
```

###### In-place operations:  
> In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.
```py
# operations that store result in operand are denoted by a _ suffix
x.copy_(y)  # copies tensor y to tensor x - shapes must match
x.t_()  # transposes x and stores back to x
tensor.add_(5)  # adds 5 to every element in tensor and stores back to tensor
```

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Datasets & DataLoaders</u>

* PyTorch provides two data primatives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` to use pre-loaded datasets as well as your own.  
* `Dataset` stores the samples and corresponding labels.  
* `DataLoader` wraps an iterable around `Dataset` to enable easy access to samples.  
* PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass torch.utils.data.Dataset and implement functions specific to the particular data. See [Image datasets](https://pytorch.org/vision/stable/datasets.html), [Text datasets](https://pytorch.org/text/stable/datasets.html), and [Audio datasets](https://pytorch.org/audio/stable/datasets.html).  

##### <u>Loading Datasets</u>  
```py
"""
Example of how to load the Fashion-MNIST dataset from TorchVision.

Fashion-MNIST is a dataset of images consisting of 60,000 training examples and 10,000 test examples.
Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.
"""
import torch  # for tensor operations and transformations
from torchvision import datasets  # for loading datasets
from torchvision.transforms import ToTensor  # for converting images to tensors

# load training dataset
training_data = datasets.FashionMNIST(
    root="data",  # path where train/test data is stored
    train=True,  # specifies training or test dataset
    download=True,  # downloads dataset from internet if not available at root
    transform=ToTensor()  # specifies the feature transformation (see below)
)

# load test dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

##### <u>Transforms</u>  
* Perform some manipulation of the data and make it suitable for training.  
* TorchVision datasets have two parameters (note these should be applied to both the training and test datasets if assigning as above) -  
  - `transform` to modify the **features**  
  - `target_transform` to modify the **labels**.  
* [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) module offers several commonly-used transforms. Below are examples of `ToTensor()` and one-hot encoding using `torchvision.transforms.Lambda` module.  
```py
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,

    # feature transformation
    transform=ToTensor(),

    # label transformation - not strictly necessary if wanting to keep original integer values
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```
* `ToTensor()` converts a PIL image or NumPy ndarray into a FloatTensor and scales the image pixel intensity values in range [0.,1.].  
* One-hot encoding is specified in the above example to the labels via `target_transform` by first createing a tensor of size 10 (number of labels in dataset), then calling `scatter_` which assigns a `value=1` on the index as given by the label `y`  

##### <u>Creating a Custom Dataset</u>  
* A custom dataset class **MUST** implement three functions: `__init__`, `__len__`, and `__getitem__`.  
* Labels are stored in a separate CSV file to the images and looks like this:  

    |Image_filename|Label|
    |:---:|:---:|
    |tshirt1.jpg|0|
    |tshirt2.jpg|0|
    |...|...|
    |ankleboot999.jpg|9|

```py
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # labels stored in a CSV file
        self.img_dir = img_dir  # images stored in directory separate to labels
        self.transform = transform  # feature transformation
        self.target_transform = target_transform  # label transformation

    def __len__(self):
        """
        :return: int, Number of samples in the dataset
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.

        :param idx: int, Index of the sample to retrieve.
        :return: tuple (image: Tensor, label: int or Tensor)
             - `image`: The image data as a tensor.
             - `label`: The label for the image, which could be an integer or tensor depending on transformations.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # sets path to directory and joins using image name from labels file
        image = read_image(img_path)  # reads image and converts to tensor
        label = self.img_labels.iloc[idx, 1]  # retrieves corresponding label from csv
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

##### <u>Iterating & Visualising the Dataset</u>  
* Can index Datasets manually like a list.  
* Use matplotlib to visualise some samples.  
```py
import matplotlib

# Set the backend for interactive plots
matplotlib.use('TkAgg')  # noqa: E402 comment here suppresses formatting warning
# or
# Set the backend for non-interactive plots (e.g., when running on a server)
# matplotlib.use('Agg')  # noqa: E402

from matplotlib import pyplot as plt

# maps numerical labels (0-9) to corresponding class names
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))  # create figure object where images will be plotted
cols, rows = 3, 3  # define layout of grid for displaying images
for i in range(1, cols * rows + 1):  # loop to create subplots for each image (range 1-10 = 9 loops)
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # creates a tensor of shape (1,), populates with random int between 0 and length of training_data, then extracts value
    img, img_label = training_data[sample_idx]  # can index datasets like a list to retrieve image and corresponding label
    figure.add_subplot(rows, cols, i)  # adds subplot to the figure to ith position of 3x3 grid
    plt.title(labels_map[img_label])  # sets title of current subplot
    plt.axis("off")  # removes axis lines and labels from subplot to show only the image
    plt.imshow(img.squeeze(),
               cmap="gray")  # displays image in subplot in grayscale. img.squeeze() removes any singleton dimensions (e.g., converting a shape of [1, 28, 28] to [28, 28]).
plt.show()  # renders entire figure with all subplots
```

##### <u>Preparing Data for Training with DataLoaders</u>  
* `Dataset` retrieves datasets features and labels one sample at a time.  
* While training a model, typically want to pass samples in "minibatches",reshuffle the data at every epoch to reduce model overfitting, and use Python's 'multiprocessing' to speed up data retrieval.  
* `DataLoader` is an iterable that abstracts this complexity in an API.  
```py
from torch.utils.data import DataLoader

# load dataset into dataloader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Get next batch of features and labels
train_features, train_labels = next(iter(train_dataloader))  

img = train_features[0].squeeze()  # gets first image in batch and removes singleton dimension
img_label = train_labels[0]  # corresponding label as tensor

# Accessing label as either original value or using map to convert
label_value = img_label
label = labels_map[img_label.item()]

# Display image
plt.imshow(img, cmap="gray")
plt.show()
```
* Each iteration of `next(iter(train_dataloader))` returns a batch of `batch_size` (64) features and labels respectfully.  
* With `shuffle=True` specified when creating DataLoader object, after iterating **all** batches, the data is shuffled.  
* Alternatively, [samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler) can be used for fine-grained control over data loading order.  

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Building a Neural Network</u>

* Comprise of layers/modules that perform operations on data.  
* The [torch.nn](https://pytorch.org/docs/stable/nn.html) namespace contains everything required to build a neural network (building blocks, e.g. layers and utilities).  
* All neural net modules subclass the [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). Your own modules (neural networks) should also subclass this class.  

```py
# typical imports required
import os  # handling file and directory paths
import torch  # tensor operations, basic operations (core PyTorch utilities), autograd functionality
from torch import nn  # classes and functions for building neural net layers and modules
from torch.utils.data import DataLoader  # load data in batches, shuffle, handle multiprocessing
from torchvision import (
    datasets,  # preloaded datasets (e.g., CIFAR-10, MNIST)
    transforms  # transforms data, usually image preprocessing and augmentation (e.g. images to tensors)
)
```

##### <u>Get Device for Training</u>  
```py
device = (
            # cuda is available on systems with NIVIDIA GPUs
            "cuda" if torch.cuda.is_available()  
            # mps is available on macOS systems that support Metal (Apple silicon e.g. M1/M2 chips)
            else "mps" if torch.backends.mps.is_available()  
            else "cpu"
        )

# Move instance of neural network model (see below) to available device (default is CPU - typically slower)
model = NeuralNetwork().to(device)

# Can also move tensors to available device 
tensor = tensor.to(device)
```

##### <u>Define the Class</u>  
* Subclass [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).  
* Initialise layers in `__init__`.  
* Operations on input data are done in the `forward` method (same for all `nn.module` subclasses). **Do NOT** directly call `model.forward()`, it is automatically called (via `nn.Module`) when we pass the model input data.  
* Move instance of NeuralNetwork to available device ([see above](#get-device-for-training)).  
* [Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) (e.g. ReLU) are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.  
* See [`torch.nn` Module](#torchnn-module) below for information on neural network **layers** and **activation functions**.  
```py
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
        logits = self.linear_relu_stack(x)
        return logits

# create instance and move to available device
# do this before training or using the model
model = NeuralNetwork().to(device)
```
* All fields inside your model object are automatically tracked by `nn.Module`, and makes all parameters accessible using your model’s `parameters()` or `named_parameters()` methods:
```py
# prints models structure
print(model)

# iterate over each parameter, and print its size and a preview of its values
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

##### <u>Using a Model</u>  
* First create a model and move to desired device.  
* Model should be pre-trained (unless you are varifying the models architecture or benchmarking its performance prior to training etc.).  
* **Do NOT** directly call `forward()` - automatically called (via [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) when passing the model input data).
```py
# create input data, typically will load in the data > see Datasets & DataLoaders above
X = torch.rand(1, 28, 28, device=device)  # [batch size, height, width], [1 image, 28 x 28]

# pass model input data. 
# Example input shape: [1,28,28] -> output shape: [batch_size, output_classes_defined_in_NeuralNetwork], [1,10]
logits = model(X)  # internally calls forward() and returns the networks logits (raw output values) as tensor

""" 
Softmax normalises the logits to a probability distribution over the classes, 
applied across the class dimension (dim=1).

pred_probab is a tensor the same shape as logits [1,10], 
but now represents probabilities instead of raw values. 
Each value is between 0 and 1 and will sum to 1 along class dimension (as specified: dim=1)
"""
pred_probab = nn.Softmax(dim=1)(logits)

# get index of the highest probability in pred_probab dim=1
y_pred = pred_probab.argmax(1)  
```

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>`torch.nn` Module</u>

[Basic building blocks](https://pytorch.org/docs/stable/nn.html) for graphs including neural net layers and activation functions:  
* [Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) quantify the difference between the predicted output and the actual target, guiding the optimisation process. See [Loss Function](#loss-functions) below for more information.  
* [Containers](https://pytorch.org/docs/stable/nn.html#containers) organize layers and operations in a modular, sequential manner.  
* [Convolution layers](https://pytorch.org/docs/stable/nn.html#convolution-layers) detect spatial hierarchies in data, commonly used in image processing for extracting features.  
* [Pooling layers](https://pytorch.org/docs/stable/nn.html#pooling-layers) reduce the spatial dimensions of the data while retaining important information, typically for down-sampling by selecting the maximum or average value in a region.  
* [Padding layers](https://pytorch.org/docs/stable/nn.html#padding-layers) adjust the spatial dimensions of data by adding extra values (typically zeros) around the edges, useful for maintaining dimensionality before and after convolutions.  
* [Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) (e.g. ReLU) are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena. See [Activation Functions](#activation-functions) below for more information.  
* [Linear layers](https://pytorch.org/docs/stable/nn.html#linear-layers) perform affine transformations (linear transformations), used to map inputs to outputs, often as the final layer in a network (aka dense layer or fully connected layer).  
* [Dropout layers](https://pytorch.org/docs/stable/nn.html#dropout-layers) helps reduce overfitting by randomly setting a fraction of input units (or elements with a specifyied probability) to zero during training.  
* [Sparce layers](https://pytorch.org/docs/stable/nn.html#sparse-layers) efficiently handle operations with sparse tensors, commonly used in large-scale data with many zeroes.  
* [Normalisation layers](https://pytorch.org/docs/stable/nn.html#normalization-layers) stabilise and accelerate training by normalising activations (across the batch or across the layer).  
* [Recurrent layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers) process sequential data by maintaining a hidden state that can capture temporal dependencies and patterns over time. Current input is influenced by previous inputs due to the use of a feedback loop. Includes long short-term memory and Elman RNN.  
* [Transformer layers](https://pytorch.org/docs/stable/nn.html#transformer-layers) handle sequence data by leveraging attention mechanisms for parallelisable processing - used in language models like GPT and BERT.  
* [Distance functions](https://pytorch.org/docs/stable/nn.html#distance-functions) measure similarity or dissimilarity between data points. `nn.PairwiseDistance` calculates Euclidean distance.  
* [Vision layers](https://pytorch.org/docs/stable/nn.html#vision-layers) contains specialised layers for image processing tasks.  
* [Shuffle layers](https://pytorch.org/docs/stable/nn.html#shuffle-layers) rearrange data order, often used to improve generalisation.  
* [DataParallel Layers](https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel) distribute computations across multiple GPUs or machines.  
* [Utilities](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) helper functions and classes for managing and manipulating models and tensors - clip parameter gradients, flatten/unflatten Module parameters to/from a single vector, fuse Modules with BatchNorm modules, convert Module parameter memory formats, apply/remove weight normalisation from Module parameters, initialise Module parameters, pruning Module parameters, parametisation.  

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Activation Functions</u>

* 

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Automatic Differentiation With Autograd</u>

* [Autograd](https://pytorch.org/docs/stable/autograd.html) keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of Function objects. In this DAG, leaf nodes are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.  
* In forward pass autograd:  
    * Runs requested operation to compute a resulting tensor.  
    * Maintains the operation’s gradient function in the DAG.  
* In backward pass (by calling `.backward()`) autograd:  
    * Computes the gradients by applying the chain rule to the operations stored in each `.grad_fn`.  
    * Accumulates them in the respective tensor’s `.grad` attribute.  
    * Propergates gradients all the way to the leaf tensors.  
* **Note** each graph is recreated from scratch after each `.backward()` call unless `retain_graph=True` is passed in call.  
* Custom [Function](https://pytorch.org/docs/stable/autograd.html#function) classes can be created to implement non-standard operations, customise gradients, optimise memory usage, customise backpropergation rules etc.  
```py
"""
A simple one-layer network with input x, parameters w, b and some loss function
We need to optimise params w and b. 
"""
import torch

x = torch.ones(5)  # input tensor (vector of 5 ones)
y = torch.zeros(3)  # expected output (vector of 3 zeros)
w = torch.randn(5, 3, requires_grad=True)  # weight matrix, initialised randomly
b = torch.randn(3, requires_grad=True)  # bias vector, initialised randomly

z = torch.matmul(x, w)+b  # linear output (logits) of the network before applying any activation function
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # computes the loss (binary cross-entropy with logits)
```
##### <u>Compute Gradients</u>
* To compute gradients of a loss function with respect to its variables (e.g. weights and biases), need to set `requires_grad` property of those tensors. Gradients will not be available for other graph nodes.  
    * Either set `requires_grad` **when creating tensor**,  
    * or later with `x.requires_grad_(True)` method.  
*  Can only perform gradient calculations using `backward()` **once on a given graph** by default, for performance reasons. If we need to do several backward calls on the same graph, we need to pass `retain_graph=True` to the `backward()` call.  
*  **Zero gradients between training iterations** to prevent accumulation from multiple backward passes, see below.  
*  Avoid in-place operations on tensors that require gradients as can cause errors during backpropergation.  
```py
# first call backward() 
loss.backward()  # calculates gradient of the loss tensor wrt graph leaves (all tensors in graph with requires_grad=true)

# then retrieve gradients from the .grad attributes
gradient_wrt_w = w.grad  
gradient_wrt_b = b.grad 

# gradients are accumulated in '.grad' attribute
# beneficial for certain optimisation algorithms,
# but require manually zeroing between training iterations to prevent accumulations from multiple backward passes
w.grad.zero()
b.grad.zero()

# alternatively 
# zeroing all parameters that an optimiser is responsible for can be done by calling `.zero_grad()` on the optimiser
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
optimiser.zero_grad()  # zeroes the gradients of all model parameters
```

##### <u>Operations & Tracking</u>  

|Item|Subheading|
|:---:|:---:|
| **1.** | [Disable gradient tracking](#disable-gradient-tracking) |
| **2.** | [Access reference to backward propergation function](#access-reference-to-backward-propergation-function) |
| **3.** | [Check leaf node tensor](#check-leaf-node-tensor) |

###### Disable gradient tracking:  
* All tensors with `requires_grad=True` are tracking their computational history and support gradient computation.  
* If only want to do forward computations (e.g. when we have trained the model and just want to apply it to some input data), you can stop tracking to save memory and computation.  
```py
z = torch.matmul(x, w)+b
print(f"tracking on: {z.requires_grad}")  # True

# surround code to turn off gradient tracking
with torch.no_grad():  
    z = torch.matmul(x, w)+b
print(f"tracking on: {z.requires_grad}")  # False

z.requires_grad_(True)  # turns tracking back on
print(f"tracking on: {z.requires_grad}")  # True


# alternatively detach a tensor from its computational graph
z_det = z.detach()  # creates a new tensor that shares same data but with tracking off
print(f"tracking on: {z_det.requires_grad}")  # False
```

###### Access reference to backward propergation function:
* A function applied to tensors to construct computational graph is an object of [Function](https://pytorch.org/docs/stable/autograd.html#function) class, which knows how to compute the function in the forwards direction, and how to compute derivative during back propergation step.  
* Reference to backward propergation function is stored in `grad_fn` - automatically created when a tensor is the result of an operation involving other tensors with `requires_grad=True`.  
```py
grad_fn_ref_z = z.grad_fn  # <AddBackward0 object at 0x00000125CBAFFE20>
grad_fn_ref_loss = loss.grad_fn  # <BinaryCrossEntropyWithLogitsBackward0 object at 0x00000125CBAFFE20> 
```

###### Check leaf node tensor:  
* Leaf tensors are tensors created by the user that have `requires_grad=True` and store their gradients in the `.grad` attribute.  
```py
print(f"w leaf tensor: {w.is_leaf}")  # True
print(f"z leaf tensor: {z.is_leaf} \n")  # False
```

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Optimising Model Parameters - Train/Test</u>

* Training is an iterative process. Each iteration, model makes a guess about output, calculates error (loss), collects the derivatives of the error wrt its parameters, and optimises these parameters using gradient decent (or other optimiser).  
* Each iteration is called an epoch which consists of two main parts:
  * The **train loop** - iterate over the training dataset and try to converge to optimal parameters.  
  * The **validation/test loop** - iterate over the test dataset to check if model performance is improving.  
* First load the [datasets and DataLoader](#datasets--dataloaders) objects - you may wish to set `batch_size` hyperparameter prior to this (to pass to the DataLoaders).  
* [Build the model](#building-a-neural-network).  
* Set the [hyperparameters](#hyperparameters).  
* Initialise a [loss function](#initialise-loss-function).  
* Define and initialise an [optimiser](#initialise-optimiser).  

##### <u>Hyperparameters</u>  
* Adjustable parameters that let you control the model optimisation process. Different hyperparameter values can impact model training and convergence rates.  
* Can use tools such as [Ray Tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html) to help find best combination of parameters.  
* Define the following hyperparameters for training:  
```py
epochs = 5  # the number times to iterate over the dataset 
batch_size = 64  # the number of data samples propagated through the network before the parameters are updated  
learning_rate = 1e-3  # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training  
```

##### <u>Initialise Loss Function</u>  
* **Loss function classes** - [torch.nn.<loss_function>](https://pytorch.org/docs/stable/nn.html#loss-functions) requires instantiation and is used as an object. They can maintain internal states or configurations that persist across calls to the instance - use when you need to maintain or reuse specific configurations across multiple invocations - **usual choice**.  
* **Stateless Loss Functions** - [torch.nn.functional.<loss_function>](https://pytorch.org/docs/stable/nn.functional.html#loss-functions) are called directly without needing to create an instance. All necessary arguments/options need to be provided in each function call - use for a more concise and straightforward implementation where state management is not required.  
* See [Loss Functions](#loss-functions) below for more information.  
```py
# example loss function class
loss_fn = torch.nn.CrossEntropyLoss()  # initialise the loss function
loss = loss_fn(predictions, targets)  # calculating loss

# example stateless loss function usage
loss = torch.nn.functional.cross_entropy(predictions, targets)  # calculating loss
```

##### <u>Initialise Optimiser</u>  
* Process of adjusting model parameters to reduce model error in each training step.  
* 

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Loss Functions</u>

* Measures the degree of dissimilarity of obtained result to the target value - want to **minimise** during training.  
* See [Initialise Loss Function](#initialise-loss-function) above for more information on initialising and use.  

|Function|Function class|For task|Notes|
|:---:|:---:|:---:|:---:|
|Mean Square Error|[nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)|Regression||
|Negative Log Likelihood|[nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)|Classification|Typically used instead of `nn.CrossEntropyLoss` when you have already applied a softmax or log-softmax operation to your model's output.|
|Cross Entropy|[nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)|Classification|Combines `nn.NLLLoss` and `nn.LogSoftmax`.<br><br>Input should be raw, unnormalised logits output by the model. These are passed through a softmax operation inside the loss function to convert them into probabilities, and then the negative log-likelihood of the correct class is computed.<br><br>Target labels should be provided as class indices (not one-hot encoded).|

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Optimisers</u>

* [torch.optim](https://pytorch.org/docs/stable/optim.html) package is for implementing various optimisation algorithms.  
* See [Initialise Optimiser](#initialise-optimiser) above for more information on initialising and use.  
* 

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  

### <u>Tensors3</u>

* 

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  