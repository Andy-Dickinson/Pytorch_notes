# <center>Pytorch Notes</center>

---  

### <center>Table of Contents</center>  
|Item|Heading|Subcontents|
|:---:|:---:|:---:|
| **1.** | [Reference Links](#reference-links) ||
| **2.** | [Tensors](#tensors) | [Initialising](#initialising-a-tensor),<br>[Attributes](#attributes-of-a-tensor),<br>[Operations](#operations-on-a-tensor) - indexing, joining, arithmetic etc., |
| **3.** | [Datasets & DataLoaders](#datasets--dataloaders) | [Loading datasets](#loading-datasets),<br>[Transforms](#transforms),<br>[Creating a Custom Dataset](#creating-a-custom-dataset),<br>[Iterating and Visualising the Dataset](#iterating-and-visualising-the-dataset),<br>[Preparing Data for Training with DataLoaders](#preparing-data-for-training-with-dataloaders) |

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

[Availiable operations](https://pytorch.org/docs/stable/torch.html) includes **arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling** (generating random values, seeding, sample from various distributions, random selection, noise addition, shuffling etc.) and more.  

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

# Alternatively
device = (
            # cuda is available on systems with NIVIDIA GPUs
            "cuda" if torch.cuda.is_available()  
            # mps is available on macOS systems that support Metal (Apple silicon e.g. M1/M2 chips)
            else "mps" if torch.backends.mps.is_available()  
            else "cpu"
        )
tensor = tensor.to(device)
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

##### <u>Iterating and Visualising the Dataset</u>  
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


### <u>Tensors3</u>

* 

<br>

[⬆ Table of Contents ⬆](#pytorch-notes)    

---  