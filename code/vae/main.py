import io
import os

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from typing import Optional, Union


# Set the backend (Option 1 or 2)
matplotlib.use('TkAgg')  # or 'Agg' if you're in a headless environment


def main():
    """
    Modify explore_latent for logging to tensorboard and optionally displaying (show)
    Different types of layers?? normalisation??

    work out how to run on HPC - check libraries using are installed

    integrate lenis code
    reduce latent space to 10 (for fine grained control and sampling)

    try changing to Beta-VAE
    """
    # CONFIG
    TRAIN = True
    USE_SEED = True
    SEED = 42
    TENSOR_BOARD_DIR = 'runs/VAE_experiment'

    # INPUT
    SINGLE_INPUT_ROWS = 28  # height in the case of an image
    SINGLE_INPUT_COLS = 28  # width

    # HYPERPARAMETERS
    LEARNING_RATE = 1e-3  # 3e-4  # Karpathy constant
    BATCH_SIZE = 100
    EPOCHS = 3  # 50

    # GENERATED IMAGES
    DIGIT_MEAN = 0
    DIGIT_VAR = 1
    SHOW_DIGIT = False
    LATENT_MEAN_R = 3
    LATENT_VAR_R = 2
    SHOW_LATENT = False

    # CHECKPOINTING
    MODEL_DIR = 'model_checkpoints'
    MODEL_FILENAME = 'checkpoint.pth'

    # set seeds for reproducibility
    if USE_SEED:
        torch.manual_seed(SEED)  # generating random numbers for CPU
        torch.cuda.manual_seed(SEED)  # generating random numbers for GPU
        torch.cuda.manual_seed_all(SEED)  # if using multiple GPUs
        np.random.seed(SEED)  # for NumPy

    # initialise tensorboard writer
    writer = SummaryWriter(log_dir=TENSOR_BOARD_DIR)

    # load datasets and dataloaders
    train_ds, test_ds, train_loader, test_loader = download_ds_make_dl(BATCH_SIZE)

    device = set_device()

    # visualise_dl(train_loader)

    model = VAE(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_epoch, best_loss = load_checkpoint(model, directory=MODEL_DIR, filename=MODEL_FILENAME, optimizer=optimizer)

    # updates optimizers learning rate if different to that of checkpoint
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE

    if TRAIN:
        for t in range(EPOCHS):
            average_loss_train, avg_recon_loss_train, avg_kl_loss_train = train(train_loader, model, optimizer, device)
            print(f"Epoch {checkpoint_epoch + t + 1} ----------------------------------------")
            print(f"Average train loss: {average_loss_train:>8f}")

            # log train metrics to tensorboard
            writer.add_scalar('Average Total Loss/Train', average_loss_train, checkpoint_epoch+t+1)
            writer.add_scalar('Average Reconstruction_Loss/Train', avg_recon_loss_train, checkpoint_epoch+t+1)
            writer.add_scalar('Average KL_Divergence/Train', avg_kl_loss_train, checkpoint_epoch+t+1)

            average_loss_test, avg_recon_loss_test, avg_kl_loss_test = test(test_loader, model, device)
            print(f"Average test loss: {average_loss_test:>8f}\n")

            # log test metrics to tensorboard
            writer.add_scalar('Average Total Loss/Test', average_loss_test, checkpoint_epoch+t+1)
            writer.add_scalar('Average Reconstruction_Loss/Test', avg_recon_loss_test, checkpoint_epoch+t+1)
            writer.add_scalar('Average KL_Divergence/Test', avg_kl_loss_test, checkpoint_epoch+t+1)

            # generate digit
            digit = generate_digit(model, DIGIT_MEAN, DIGIT_VAR, height=SINGLE_INPUT_ROWS, width=SINGLE_INPUT_COLS, show=SHOW_DIGIT)

            # log reconstructed image to tensorboard to monitor the quality of the decoder over time
            writer.add_image(f'Generated_digit/epoch_{checkpoint_epoch+t+1}/mean_{DIGIT_MEAN}/var_{DIGIT_VAR}', digit.unsqueeze(0), checkpoint_epoch+t+1)

            # log latent space image to tensorboard to monitor over time
            # The mean values typically represent the center of the learned latent distribution for each latent variable. A good range for visualising the latent space is between -3 and 3. This is based on the standard normal distribution from which the VAE samples latent variables.
            # The variance indicates the spread of the latent distribution. In most cases, the variance is positive and centered around 1 for a standard normal distribution, so visualising between 0.5 and 2 can be useful. Higher variance means more uncertainty in that area of the latent space, while lower variance indicates more certainty.
            # early epochs - use variance 1.5-2 to capture uncertainty, mid epochs - latent space starts clustering, use variance around 1, later epochs - latent space should be more organised, use variance 0.5-1 to better visualise structure
            latent_space_fig_arr = plot_latent_space(model, LATENT_MEAN_R, LATENT_VAR_R, img_height=SINGLE_INPUT_ROWS, img_width=SINGLE_INPUT_COLS, show=SHOW_LATENT)
            writer.add_image(f'Latent_space/epoch_{checkpoint_epoch+t+1}', latent_space_fig_arr, checkpoint_epoch+t+1)

            # saves model each time loss is bettered
            if best_loss is None or average_loss_test < best_loss:
                best_loss = average_loss_test
                save_checkpoint(model, checkpoint_epoch + t + 1, best_loss, directory=MODEL_DIR, filename=MODEL_FILENAME, optimizer=optimizer)
        print("Done!")
    else:
        explore_latent(model, device=device)

        # # generate digit
        generate_digit(model, DIGIT_MEAN, DIGIT_VAR, height=SINGLE_INPUT_ROWS, width=SINGLE_INPUT_COLS, show=SHOW_DIGIT)
        # # generate latent space
        plot_latent_space(model, LATENT_MEAN_R, LATENT_VAR_R, img_height=SINGLE_INPUT_ROWS, img_width=SINGLE_INPUT_COLS, show=SHOW_LATENT)

    # close tensorboard writer
    writer.close()


def set_device() -> torch.device:
    """
    Sets device based on availability.

    :return: torch.device object set to ("cuda", "mps", or "cpu")
    """
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def explore_latent(model: torch.nn.Module, latent_dim: int = 10, n: int = 25, img_height: int = 28, img_width: int = 28, figsize: int = 15, device: torch.device = set_device()) -> None:
    """
    Plots latent space by varying each dimension between -3 and 3.
    Latent vector is sampled from standard normal distribution (mean 0, variance 1).
    This vector is cloned before each change, so vector is fixed and it is only that dimension which is varied.

    :param model: Model to use to reconstruct image. Must contain reconstruct method which takes a 1D vector of size latent_dim
    :param latent_dim: Number of elements in latent dimension vector
    :param n: Number of images to plot along y-axis
    :param img_height: Height of each image
    :param img_width: Width of each image
    :param figsize: Overall size of canvas
    :param device: Device to use for tensor operations. If no argument passed, set_device() is called
    :return: None
    """
    model.eval()

    # random latent vector sampled from standard normal distribution (mean 0, variance 1)
    z = torch.randn(latent_dim).to(device)

    # create a blank canvas to hold the entire grid of images
    figure = np.zeros((n * img_height, latent_dim * img_width), dtype=np.float32)

    # create a grid for varying the latent dimensions
    grid_x = np.arange(latent_dim)  # latent dimension indices for x-axis (0 to latent_dim-1)
    grid_y = np.linspace(-3, 3, n)[::-1]  # values for modifying the latent vector in y-axis

    # round the grid values to 2 decimal places
    grid_y = np.round(grid_y, 2)

    for i, yi in enumerate(grid_y):
        for j in range(latent_dim):  # iterate over latent dimensions
            z_mod = z.clone()
            z_mod[j] = yi  # modify one dimension of cloned latent vector
            recon_x = model.reconstruct_from_vector(z_mod)
            digit = recon_x.detach().cpu().reshape(img_height, img_width)  # reshape vector to 2d array
            figure[i * img_height: (i + 1) * img_height, j * img_width: (j + 1) * img_width] = digit  # place on canvas

            # record mean and variance
            mean_new = z_mod.mean()
            var_new = z_mod.var()

    # set overall figsize and title
    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Exploration\nVarying dimension values between -3 and 3')
    plt.imshow(figure, cmap="Greys_r")

    # set tick positions and labels
    pixel_range_x = set_pixel_range(latent_dim, img_width)  # define where the ticks should appear in relation to images
    plt.xticks(pixel_range_x, grid_x)
    plt.xlabel("Dimension")

    pixel_range_y = set_pixel_range(n, img_height)
    plt.yticks(pixel_range_y, grid_y)
    plt.ylabel("Value")

    plt.show()
    plt.close()


def save_checkpoint(model: torch.nn.Module, epoch: int, loss: float, directory: str = 'model_checkpoints', filename: str = 'checkpoint.pth', optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """
    Saves model checkpoint.
    Creates directory if does not exist.
    Overwrites file if already exists.
    Saves:
        Model state dict,
        Epochs run for model,
        Loss of model,
        Optionally, optimizer state dict.

    :param model: Model to save
    :param epoch: Epochs model has run
    :param loss: Best loss (or other loss to save)
    :param directory: Directory to save file in (creates if does not exist)
    :param filename: Filename of checkpoint
    :param optimizer: Optional, optimizer to save
    :return: None
    """
    # creates directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # using torch.save to write dictionary object to file
    torch.save(checkpoint, filename)
    print(f"checkpoint saved, epoch: {epoch}, best loss: {loss}\n")


def load_checkpoint(model: torch.nn.Module, directory: str = 'model_checkpoints', filename: str = 'checkpoint.pth', optimizer: Optional[torch.optim.Optimizer] = None) -> tuple[int, Optional
[float]]:
    """
    Loads a checkpoint (weights only) includes:
        models state dict,
        epoch model was saved at (0 if not saved),
        the loss of the model (None if not saved),
        Optionally optimizers state dict.

    :param model: Instance of model to load state into
    :param directory: Dir to look for checkpoint in
    :param filename: Filename to load
    :param optimizer: Optional, to load optimizer state dict into
    :return: epoch (0 if not found), loss (None if not found)
    """
    filepath = os.path.join(directory, filename)

    # initialise epoch and loss
    epoch = 0
    loss = None

    # check if directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.\nEpochs run for model set to {epoch}, best loss set to {loss}.\n")
        return epoch, loss

    # check if the checkpoint file exists
    if not os.path.isfile(filepath):
        print(f"No checkpoint file '{filename}' found in directory '{directory}'.\nEpochs run for model set to {epoch}, best loss set to {loss}.\n")
        return epoch, loss

    # try to load the checkpoint
    try:
        checkpoint = torch.load(filepath, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Optimizer state loaded for optimizer {optimizer}.")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)

        print(f"Checkpoint loaded with best loss: {loss} after {epoch} epochs\n")
    except Exception as e:
        print(f"An error occurred while loading the checkpoint: {e}.\nEpochs run for model set to {epoch}, best loss set to {loss}.\n")

    return epoch, loss


class Encoder(nn.Module):
    """
    Encodes input to a latent space representation with latent dim dimensions.
    Encoding is in the form mean and log variance tensors with shapes (batch size, latent dim)
    """
    def __init__(self, input_dim: int = 784, latent_dim: int = 10) -> None:
        """
        Constructor defining Encoder model architecture.

        :param input_dim: Flattened input dimension
        :param latent_dim: Latent space vector dimension
        """
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(input_dim, 400)
        self.hidden_layer = nn.Linear(400, 200)
        self.hidden_layer2 = nn.Linear(200, 100)
        self.hidden_layer3 = nn.Linear(100, 50)

        # latent mean and variance layers
        self.mean_layer = nn.Linear(50, latent_dim)  # μ represents center of the Gaussian distribution for each latent variable
        # logvar_layer outputs the log-variance (log(σ^2)) of the Gaussian distribution for each latent variable
        # using log-variance rather than variance directly ensures numerical stability, as variance must be positive.
        # logvar_layer can produce negative values, but they are transformed back into positive variances during reparameterisation.
        self.logvar_layer = nn.Linear(50, latent_dim)  # log(σ^2) represents the uncertainty or spread of the Gaussian distribution for each latent variable

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Encoder (z|x).
        Mean and log variance tensors have shape of (batch size, latent dim).

        :param x: Input tensor with shape (batch size, input dim)
        :return: mean tensor, log variance tensor with shapes (batch size, latent dim)
        """
        h = self.LeakyReLU(self.input_layer(x))
        h2 = self.LeakyReLU(self.hidden_layer(h))
        h3 = self.LeakyReLU(self.hidden_layer2(h2))
        h4 = self.LeakyReLU(self.hidden_layer3(h3))

        mean, logvar = self.mean_layer(h4), self.logvar_layer(h4)

        return mean, logvar


class Decoder(nn.Module):
    """
    Decodes a latent vector into a reconstructed output (x|z).
    """
    def __init__(self, output_dim: int = 784, latent_dim: int = 10) -> None:
        """
        Constructor defining Decoder model architecture.

        :param output_dim: Reconstructed dimension
        :param latent_dim: Latent space vector dimension
        """
        super(Decoder, self).__init__()

        self.hidden_layer = nn.Linear(latent_dim, 50)
        self.hidden_layer2 = nn.Linear(50, 100)
        self.hidden_layer3 = nn.Linear(100, 200)
        self.hidden_layer4 = nn.Linear(200, 400)
        self.output_layer = nn.Linear(400, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder (x|z).

        :param z: Latent vector with shape (batch size, latent dim)
        :return: (x|z) with shape (batch size, input dim)
        """
        h = self.LeakyReLU(self.hidden_layer(z))
        h2 = self.LeakyReLU(self.hidden_layer2(h))
        h3 = self.LeakyReLU(self.hidden_layer3(h2))
        h4 = self.LeakyReLU(self.hidden_layer4(h3))

        recon_x = torch.sigmoid(self.output_layer(h4))  # output values in range [0,1]

        return recon_x


class VAE(nn.Module):
    """
    Integrates both Encoder and Decoder classes.
    Manages the reparameterisation operation.
    Contains reconstruct method to reconstruct x given a mean and log variance value.
    """
    def __init__(self, input_dim: int = 784, latent_dim: int = 10, device: Optional[torch.device] = None) -> None:
        """
        Constructor for VAE model.
        Map inputs to a latent space mean and log variance vectors.
        Uses reparameterisation trick to sample from the latent normal distribution with mean 0 and variance 1.
        This allows sampling from Gaussian distribution in the latent space in a way that allows gradients to flow through.

        :param input_dim: Flattened input dimension size
        :param latent_dim: Latent space dimension
        :param device: Device to use, if none is passed, calls set_device()
        """
        super(VAE, self).__init__()
        self.device = device if device is not None else set_device()
        self.latent_dim = latent_dim

        # encoder layer
        self.encoder = Encoder(input_dim, self.latent_dim)

        # decoder layer
        self.decoder = Decoder(input_dim, self.latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input into mean and log variance of the latent space distribution.

        :param x: Input tensor with shape (batch size, input dim)
        :return: Mean and log variance tensors with shape (batch size, latent dim)
        """
        mean, logvar = self.encoder(x)

        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent vector into a reconstructed output.

        :param z: Latent vector with shape (batch size, latent dim)
        :return: (x|z) with shape (batch size, input dim)
        """
        recon_x = self.decoder(z)
        return recon_x

    def reparameterisation(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterisation trick to sample from the latent normal distribution with mean 0 and variance 1.
        Allows sampling from Gaussian distribution in the latent space in a way that allows gradients to flow through.

        :param mean: Mean of the latent space distribution
        :param logvar: Log variance of the latent space distribution
        :return: Sampled latent vector with shape (batch size, latent dim)
        """
        # epsilon actually reparameterises our VAE network. This allows the mean and log-variance vectors to still remain as the learnable parameters of the network while still maintaining the stochasticity of the entire system
        epsilon = torch.randn_like(logvar).to(self.device)  # sample noise from a standard normal distribution with mean 0 and variance 1. Values can be negative or positive. Tensor will have same shape as logvar
        std = torch.exp(0.5 * logvar)  # compute standard deviation (σ) and ensures it is always positive
        z = mean + std * epsilon  # reparameterisation trick to allow backpropagation through stochastic sampling

        return z

    def reconstruct(self, mean: float = 0., logvar: float = 1.) -> torch.Tensor:
        """
        Reconstruct x using a single mean and variance value across the entire latent space.

        :param mean: Mean
        :param logvar: Log variance
        :return: (x|z) with shape (batch size, input dim)
        """
        # convert inputs to tensors and ensure they are on the correct device
        mean = torch.tensor(mean).to(self.device)
        logvar = torch.tensor(logvar).to(self.device)

        # calculate standard deviation from log-variance
        std = torch.exp(0.5 * logvar)

        # sample a latent vector from a standard normal distribution (values can be negative)
        latent_vector = torch.randn(1, self.latent_dim).to(self.device) * std + mean

        # reconstruct using the generated latent vector
        recon_x = self.decode(latent_vector)

        return recon_x

    def reconstruct_from_vector(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct x using a fixed latent_vector.

        :param latent_vector: Fixed latent vector of shape (latent_dim)
        :return: (x|z) with shape (batch size, input dim)
        """
        if latent_vector.dim() == 1 and latent_vector.shape[0] == self.latent_dim:
            # convert vector to include batch dim for decoding
            latent_vector = latent_vector.unsqueeze(0).to(self.device)

            # reconstruct using the generated latent vector
            recon_x = self.decode(latent_vector)

            return recon_x
        else:
            print(f"Latent vector has shape: {latent_vector.shape}. Should have shape: ({self.latent_dim})")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE model (z|x) > mean & log variance tensors > parameterised > (x|z).
        Reconstructed x has shape (batch size, input dim).
        Mean and log variance tensors have shape of (batch size, latent dim).

        :param x: Input tensor with shape (batch size, input dim)
        :return: (x|z), mean tensor, log variance tensor
        """
        mean, logvar = self.encode(x)
        z = self.reparameterisation(mean, logvar)
        recon_x = self.decode(z)

        return recon_x, mean, logvar


def loss_function(x: torch.Tensor, recon_x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates reconstruction loss using binary_cross_entropy and KL Divergence loss.

    :param x: Flattened inputs (x)
    :param recon_x: Reconstructed (x|z)
    :param mean: Mean tensor
    :param logvar: Log variance tensor
    :return: (reconstruction loss, KL divergence loss)
    """
    # pixel values are treated as probabilities, BCE is appropriate for measuring how well the reconstructed image (recon_x) matches the original image (x) on a per-pixel basis
    reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # regularisation loss
    # KL(q(z∣x) ∥ p(z))= −1/2 ∑(1 + log(σ^2) − μ^2 − σ^2) where μ and σ^2 are the mean and variance of the latent distribution q(z∣x), respectively
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # difference between the learned latent distribution (mean and variance) and the prior distribution (typically a standard normal distribution - with mean 0 and variance 1)

    return reconstruction_loss, KLD


def train(train_loader: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device = set_device()) -> tuple[float, float, float]:
    """
    Trains model passed in batches set in DataLoader.
    Calculates loss using loss_function which should return reconstruction loss and KL divergence.

    :param train_loader: Input data
    :param model: Torch model
    :param optimizer: Optimizer to use
    :param device: Device to use, if none passed, calls set_device()
    :return: (average loss, average reconstruction loss, average kl loss)
    """
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    processed = 0

    for batch_idx, (inputs, _) in enumerate(tqdm(train_loader, desc="Batches", unit="bat")):
        # flatten the inputs based on their shape (from 28x28 to a vector of size 784 (MNIST))
        batch_size = inputs.size(0)
        num_features = torch.prod(torch.tensor(inputs.shape[1:])).item()
        inputs_flattened = inputs.view(batch_size, num_features).to(device)

        optimizer.zero_grad()

        recon_x, mean, logvar = model(inputs_flattened)

        recon_loss, kl_loss = loss_function(inputs_flattened, recon_x, mean, logvar)
        loss = recon_loss + kl_loss

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        processed += batch_size

        loss.backward()
        optimizer.step()

    average_loss = total_loss / processed
    avg_recon_loss = total_recon_loss / processed
    avg_kl_loss = total_kl_loss / processed

    return average_loss, avg_recon_loss, avg_kl_loss


def test(test_loader: DataLoader, model: torch.nn.Module, device: torch.device = set_device()) -> tuple[float, float, float]:
    """
    Tests model passed in batches set in DataLoader.
    Calculates loss using loss_function which should return reconstruction loss and KL divergence.

    :param test_loader: Input data
    :param model: Torch model
    :param device: Device to use, if none passed, calls set_device()
    :return: (average loss, average reconstruction loss, average kl loss)
    """
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    processed = 0

    with torch.no_grad():  # No need for gradients during testing
        for inputs, _ in tqdm(test_loader, desc="Batches", unit="bat"):
            batch_size = inputs.size(0)
            num_features = torch.prod(torch.tensor(inputs.shape[1:])).item()
            inputs_flattened = inputs.view(batch_size, num_features).to(device)

            recon_x, mean, logvar = model(inputs_flattened)

            recon_loss, kl_loss = loss_function(inputs_flattened, recon_x, mean, logvar)
            loss = recon_loss + kl_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            processed += batch_size

        average_loss = total_loss / processed
        avg_recon_loss = total_recon_loss / processed
        avg_kl_loss = total_kl_loss / processed

    return average_loss, avg_recon_loss, avg_kl_loss


def download_ds_make_dl(batch_size: int = 100) -> tuple[MNIST, MNIST, DataLoader, DataLoader]:
    """
    Downloads MINST dataset as train/test datasets and wraps into train/test DataLoaders.

    :param batch_size: DataLoader batch sizes
    :return: (train dataset, test dataset, train loader, test loader)
    """
    # create a transform to apply to each datapoint
    transform = transforms.Compose([transforms.ToTensor()])

    # download the MNIST datasets
    path = "data"
    train_dataset = MNIST(root=path, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=path, train=False, transform=transform, download=True)

    # create train and test dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader


def visualise_dl(train_loader: DataLoader, samples: int = 25, figsize: tuple[int, int] = (5, 5)) -> None:
    """
    Displays first number of samples from DataLoader.

    :param train_loader: Input data
    :param samples: Number of samples to display
    :param figsize: Overall size of output plot
    :return: None
    """
    dataiter = iter(train_loader)
    image = next(dataiter)
    num_samples = len(image[0]) if samples is None or samples <= 0 else min(samples, len(image[0]))
    sample_images = [image[0][i, 0] for i in range(num_samples)]

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=figsize, axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        ax.imshow(im, cmap='gray')
        ax.axis('off')
    plt.show()


def generate_digit(model: torch.nn.Module, mean: float, logvar: float, height: int = 28, width: int = 28, show: bool = True) -> torch.Tensor:
    """
    Reconstructs a digit using the passed models reconstruct method passing in a mean and log variance value.

    :param model: Torch model. Must contain a reconstruct method that takes mean and log variance arguments
    :param mean: Mean value to pass to the model
    :param logvar: Log variance to pass to the model
    :param height: Height of reconstructed image
    :param width: Width of reconstructed image
    :param show: Display reconstructed image
    :return: Reconstructed image
    """
    model.eval()
    x_decoded = model.reconstruct(mean, logvar)  # Decode to get the generated digit
    digit = x_decoded.detach().cpu().reshape(height, width)  # reshape vector to 2d array

    if show is True:
        plt.imshow(digit, cmap='gray')
        plt.axis('off')
        plt.show()

    plt.close()
    return digit


def plot_latent_space(model: torch.nn.Module, mean_range: Union[int, float, tuple[Union[int, float], Union[int, float]]] = (-1., 1.), logvar: Union[int, float, tuple[Union[int, float], Union[int, float]]] = 1., n: int = 25, img_height: int = 28, img_width: int = 28, figsize: int = 15, show: bool = True) -> np.ndarray:
    """
    Plots latent space of model for specified range of mean and log variance.
    Displays image if show = True.
    Returns image which includes axes and title as numpy array (CHW). i.e. rendered as graphics.

    :param model: Model to use. Must have regenerate(mean, logvar) method
    :param mean_range: Mean range to display. If int or float, displays negative to positive range
    :param logvar: Log variance range to display. If int or float, displays negative to positive range
    :param n: Number of images along EACH axis, e.g. 25 = grid with 25x25 images = 625 total
    :param img_height: Number of pixels of each image height in latent space
    :param img_width: Number of pixels of each image width in latent space
    :param figsize: Overall size of output plot - note if height, width and n cause plot to be larger than figsize, may lead to distortion or compression issues
    :param show: Displays plot
    :return: Latent space image as numpy array (CHW)
    """
    mean_range = format_range(mean_range)
    logvar = format_range(logvar)

    model.eval()

    # define canvas size (where images are placed)
    # n is number of images in the grid
    figure = np.zeros((img_height * n, img_width * n), dtype=np.float32)  # ensure values in canvas are floats, required for later operations

    # define the sampling points in the latent space, n points equally spaced between mean and variance ranges
    grid_x = np.linspace(mean_range[0], mean_range[1], n)
    grid_y = np.linspace(logvar[0], logvar[1], n)[::-1]  # inverts order so top of grid represents higher values

    # round the grid values to 2 decimal places
    grid_x = np.round(grid_x, 2)
    grid_y = np.round(grid_y, 2)

    # for each (xi, yi) in latent space, a sample is created using model
    # this is resized to fit the canvas and placed in appropriate position in figure
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            x_decoded = model.reconstruct(xi, yi)  # Decode to get the generated digit
            digit = x_decoded.detach().cpu().reshape(img_height, img_width)  # reshape vector to 2d array
            figure[i * img_height: (i + 1) * img_height, j * img_width: (j + 1) * img_width] = digit

    # create a figure and an axes object
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(figure, cmap="Greys_r")

    # set title
    plt.title(f'VAE Latent Space Visualisation\nMean Range: {mean_range}, Variance: {logvar}')

    # Set the tick positions and labels for axes
    pixel_range_x = set_pixel_range(n, img_width)  # define where the ticks should appear in relation to images
    ax.set_xticks(pixel_range_x)
    ax.set_xticklabels(grid_x)
    ax.set_xlabel("Mean")

    pixel_range_y = set_pixel_range(n, img_height)  # define where the ticks should appear in relation to images
    ax.set_yticks(pixel_range_y)
    ax.set_yticklabels(grid_y)
    ax.set_ylabel("Var")

    # save the figure to a buffer
    buf = io.BytesIO()  # in-memory binary stream
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)  # removes any extra whitespace (bboc), removes any padding (pad)
    buf.seek(0)  # rewinds buffer to the beginning, so it can be read from start
    buf_image = Image.open(buf).convert('RGB')  # opens image using PIL and convert to RGB (essential for tensorboard)

    if show is True:
        buf_image.show()

    np_img = np.array(buf_image)  # converts PIL image to numpy array
    np_img = np_img.transpose((2, 0, 1))  # convert from HWC to CHW format for tensorboard

    plt.close(fig)
    buf.close()

    return np_img


def set_pixel_range(n: int, img_size: int) -> np.ndarray:
    """
    Sets pixel range for plotting multiple images and aligning tick marks with center of images.

    :param n: number of images to be plotted
    :param img_size: number of pixels per image along axis
    :return: pixel range for axis
    """
    start_range = img_size // 2  # allows first tick mark to align with center of first image in the grid
    end_range = n * img_size + start_range  # allows last tick mark to align with center of last image in grid
    pixel_range = np.arange(start_range, end_range, img_size)  # defines where tick marks will be placed along axis
    return pixel_range


def format_range(value: Union[int, float, tuple[Union[int, float], Union[int, float]]]) -> tuple[float, float]:
    """
    Formats passed value as tuple of floats of length 2.

    :param value: int, float or tuple of int or floats
    :return: (tuple) If tuple arg, ensures len=2 and converts to float tuple. If int or float, returns negative to positive float as tuple
    """
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Tuple must have exactly 2 elements.")
        return float(value[0]), float(value[1])
    elif isinstance(value, (int, float)):
        return -float(value), float(value)
    else:
        raise ValueError("Input must be an int, float, or a tuple of two ints/floats.")


if __name__ == '__main__':
    main()
