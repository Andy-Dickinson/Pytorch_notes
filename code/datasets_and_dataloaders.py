"""
*******************************
Example of how to load the Fashion-MNIST dataset from TorchVision.

Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples.
Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.
*******************************
"""
import torch  # for tensor operations and transformations
from torchvision import datasets  # for loading datasets
from torchvision.transforms import ToTensor, Lambda  # for converting images to tensors

# load training dataset
training_data = datasets.FashionMNIST(
    root="data",  # path where train/test data is stored
    train=True,  # specifies training or test dataset
    download=True,  # downloads dataset from internet if not available at root
    transform=ToTensor()  # specifies the feature transformation,
                          # here the FashionMNST features are in PIL image format and the labels are integers.
                          # ToTensor() converts a PIL image or NumPy ndarray into a FloatTensor and scales
                          # the image pixel intensity values in range [0.,1.]
    # can one-hot encode labels as tensor instead of keeping original integer value:
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                          # target_transform specifies the label transformation,
                          # here, first creates a tensor of size 10 (number of labels in dataset)
                          # then calls scatter_ which assigns a value=1 on the index as given by the label y (one hot encoding)
)

# load test dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

"""
*******************************
Iterating and Visualising the Dataset
*******************************
"""
import matplotlib

# For interactive plots
matplotlib.use('TkAgg')  # noqa: E402 - suppresses formatting warning.
# or
# For non-interactive plots (e.g., when running on a server)
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
    sample_idx = torch.randint(len(training_data), size=(
        1,)).item()  # creates a tensor of shape (1,), populates with random int between 0 and length of training_data, then extracts value
    img, img_label = training_data[
        sample_idx]  # can index datasets like a list to retrieve image and corresponding label
    figure.add_subplot(rows, cols, i)  # adds subplot to the figure to ith position of 3x3 grid
    plt.title(labels_map[img_label])  # sets title of current subplot
    plt.axis("off")  # removes axis lines and labels from subplot to show only the image
    plt.imshow(img.squeeze(),
               cmap="gray")  # displays image in subplot in grayscale. img.squeeze() removes any singleton dimensions (e.g., converting a shape of [1, 28, 28] to [28, 28]).
plt.show()  # renders entire figure with all subplots

"""
*******************************
Creating a Custom Dataset
*******************************
"""
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    A custom dataset class MUST implement three functions: __init__, __len__, and __getitem__

    Labels are stored in a separate CSV file to the images and looks like this:
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ......
        ankleboot999.jpg, 9
    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # labels stored in a CSV file
        self.img_dir = img_dir  # images stored in directory separate to labels
        self.transform = transform
        self.target_transform = target_transform

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
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[
            idx, 0])  # sets path to directory and joins using image name from labels file
        image = read_image(img_path)  # reads image and converts to tensor
        label = self.img_labels.iloc[idx, 1]  # retrieves corresponding label from csv
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


"""
*******************************
Preparing data for training with DataLoaders

'Dataset' retrieves datasets features and labels one sample at a time.
While training a model, typically want to pass samples in "minibatches",
reshuffle the data at every epoch to reduce model overfitting,
and use Python's 'multiprocessing' to speed up data retrieval.
'DataLoader' is an iterable that abstracts this complexity in an API.
*******************************
"""
from torch.utils.data import DataLoader

# load dataset into dataloader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
"""
Each iteration of below returns a batch of 'batch_size' (64) features and labels respectfully.
With shuffle=True specified when creating dataloader object, after iterating all batches, the data is shuffled.
Alternatively, samplers can be used for fine-grained control over data loading order. See https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
"""
train_features, train_labels = next(iter(train_dataloader))  # returns next batch of features and labels
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()  # gets first image in batch and removes singleton dimension
img_label = train_labels[0]  # corresponding label as tensor
print(f"Label: {img_label} = {labels_map[img_label.item()]}")
plt.imshow(img, cmap="gray")
plt.show()
