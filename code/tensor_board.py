import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

"""
Ensure both tensorboard and tensorflow are installed with pip

Open a terminal (in relevant virtual environment - 
        # On Windows
        path\to\your\venv\Scripts\activate
        # On macOS/Linux
        source path/to/your/venv/bin/activate
        )
        
Run TensorBoard and point it to the directory where your logs are saved (e.g., runs):
tensorboard --logdir=<path/to/runs/directory>
By default, TensorBoard runs on port 6006. If you need to specify a different port, use:
tensorboard --logdir=<path/to/runs/directory> --port=XXXX

Open your web browser and navigate to:
http://localhost:6006
'"""

training_data = datasets.FashionMNIST(
    # Load datasets and DataLoaders,Define model and move instance to device
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


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

nn_model = NeuralNetwork().to(device)

# define hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# initialize TensorBoard SummaryWriter
# writer will output to ./runs/ directory by default - no arguments required
writer = SummaryWriter('runs/experiment_name')

def train_loop(train_loader, model, loss_fn, optimizer, epoch):
    ds_size = len(train_loader.dataset)
    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Compute prediction and loss
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # every 100 batches, log current loss and progress
        if batch_idx % 100 == 0:
            loss, samples_processed = loss.item(), (batch_idx + 1) * batch_size
            print(f"loss: {loss:>7f}  [{samples_processed:>5d}/{ds_size:>5d}]")

            # log loss to TensorBoard
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)

def test_loop(test_loader, model, loss_fn, epoch):
    model.eval()

    ds_size = len(test_loader.dataset)
    num_batches = len(test_loader)

    total_test_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            pred = model(inputs)
            total_test_loss += loss_fn(pred, labels).item()

            # number of correct predictions
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    total_test_loss /= num_batches
    accuracy = 100 * (correct / ds_size)
    print(f"Test Error: \n Accuracy: {accuracy:>.1f}%, Avg loss: {total_test_loss:>.8f} \n")

    # Log test loss and accuracy to TensorBoard
    writer.add_scalar('Loss/test', total_test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

# define loss function and optimizer
ce_loss = nn.CrossEntropyLoss()
sgd_optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, nn_model, ce_loss, sgd_optimizer, t)
    test_loop(test_dataloader, nn_model, ce_loss, t)

# close TensorBoard writer
writer.close()
print("Done!")
