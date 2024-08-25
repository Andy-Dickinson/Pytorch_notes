import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
*******************************************
Prerequisite Code:
Load datasets and DataLoaders,
Define model and move instance to device
"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, inputs):
        inputs_flat = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs_flat)
        return logits


device = (
    # cuda is available on systems with NVIDIA GPUs
    "cuda" if torch.cuda.is_available()
    # mps is available on macOS systems that support Metal (Apple silicon e.g. M1/M2 chips)
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

nn_model = NeuralNetwork().to(device)
"""
*******************************************
"""

# define hyperparameters
learning_rate = 1e-3
batch_size = 64  # would be better before defining DataLoaders above
epochs = 5


def train_loop(train_loader, model, loss_fn, optimizer):
    ds_size = len(train_loader.dataset)  # total number of samples in training dataset

    # Set the model to training mode - important for batch_idx normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # iterates training data in batches
    # for each batch loop, makes a prediction, calculates loss, optimises the parameters
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Compute prediction and loss
        pred = model(inputs)  # forward pass
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()  # compute gradients of loss wrt model parameters
        optimizer.step()  # update model parameters based on computed gradients
        optimizer.zero_grad()  # reset gradients to prevent accumulating across batches

        # every 100 batches, log current loss and progress
        if batch_idx % 100 == 0:
            loss, samples_processed = loss.item(), (batch_idx + 1) * batch_size
            print(f"loss: {loss:>7f}  [{samples_processed:>5d}/{ds_size:>5d}]")


def test_loop(test_loader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    ds_size = len(test_loader.dataset)  # total number of samples in test dataset
    num_batches = len(test_loader)

    # initialise cumulative total test loss and number of correct predictions
    total_test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        # iterates test data in batches
        for inputs, labels in test_loader:
            pred = model(inputs)  # forward pass, pred.shape = [64, 10] = [samples_in_batch, class_scores-logits_for_each_sample]
            total_test_loss += loss_fn(pred, labels).item()  # calculates loss for batch and accumulates it in total_test_loss

            # pred.argmax(1) gives the index of the maximum logit (class prediction) for each sample
            # pred.argmax(1).shape = [64] = number of samples in the batch
            # (pred.argmax(1) == labels) compares predicted class indices with true labels -> boolean tensor
            # (pred.argmax(1) == labels).type(torch.float) converts boolean tensor to float (1.0 for correct, 0.0 for incorrect)
            # .sum() adds up all the 1.0 values, counting the total number of correct predictions
            # .item() converts the result to a Python float
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()  # number of correct predictions

    total_test_loss /= num_batches  # average loss across all batches
    correct /= ds_size  # accuracy = number_of_correct_predictions / total_number_of_samples
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_test_loss:>8f} \n")  # accuracy displayed as percentage


# define loss function and optimizer
ce_loss = nn.CrossEntropyLoss()
sgd_optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

# iterate train/test loops in epochs
# after each epoch, DataLoader will shuffle the samples if shuffle=True was defined which helps reduce overfitting
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, nn_model, ce_loss, sgd_optimizer)
    test_loop(test_dataloader, nn_model, ce_loss)
print("Done!")
