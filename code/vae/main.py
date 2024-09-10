import io
import os

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from typing import Optional, Union


# Set the backend (Option 1 or 2)
matplotlib.use('TkAgg')  # or 'Agg' if you're in a headless environment


def main():
    """
    Add config and set CONSTANT params
    add doc comments
    add progression bar with tqdm

    work out how to run on HPC - check libraries using are installed

    integrate lenis code
    reduce latent space to 10 (for fine grained control and sampling)
    """
    # set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)  # generating random numbers for CPU
    torch.cuda.manual_seed(seed)  # generating random numbers for GPU
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)  # for NumPy

    # initialise tensorboard writer
    writer = SummaryWriter(log_dir='runs/VAE_experiment')

    # define hyperparameters
    learning_rate = 1e-3  # 3e-4  # Karpathy constant
    batch_size = 100
    epochs = 1  # 50

    # load datasets and dataloaders
    train_ds, test_ds, train_loader, test_loader = download_ds_make_dl(batch_size)

    device = set_device()

    # visualise_dl(train_loader)

    model = VAE(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_epoch, best_loss = load_checkpoint(model, optimizer=optimizer)

    # updates optimizers learning rate if different to that of checkpoint
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    generate_digit(model, 1, 1, show=True)
    plot_latent_space(model, 3, 1, show=True)

    exit()

    for t in range(epochs):
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

        # generate digit with mean 0 and variance 1
        mean = 0.0
        var = 1.0
        digit = generate_digit(model, mean, var, show=False)

        # log reconstructed image to tensorboard to monitor the quality of the decoder over time
        writer.add_image(f'Generated_digit/epoch_{checkpoint_epoch+t+1}/mean_{mean}/var_{var}', digit.unsqueeze(0), checkpoint_epoch+t+1)

        # log latent space image to tensorboard to monitor over time
        # The mean values typically represent the center of the learned latent distribution for each latent variable. A good range for visualising the latent space is between -3 and 3. This is based on the standard normal distribution from which the VAE samples latent variables.
        # The variance indicates the spread of the latent distribution. In most cases, the variance is positive and centered around 1 for a standard normal distribution, so visualising between 0.5 and 2 can be useful. Higher variance means more uncertainty in that area of the latent space, while lower variance indicates more certainty.
        # early epochs - use variance 1.5-2 to capture uncertainty, mid epochs - latent space starts clustering, use variance around 1, later epochs - latent space should be more organised, use variance 0.5-1 to better visualise structure
        latent_space_fig_arr = plot_latent_space(model, 3, 2, show=False)
        writer.add_image(f'Latent_space/epoch_{checkpoint_epoch+t+1}', latent_space_fig_arr, checkpoint_epoch+t+1)

        # saves model each time loss is bettered
        if best_loss is None or average_loss_test < best_loss:
            best_loss = average_loss_test
            save_checkpoint(model, checkpoint_epoch + t + 1, best_loss, optimizer=optimizer)
    print("Done!")

    # generate a single digit using the decoder passing in mean and variance values
    # generate_digit(model, 0.0, 1.0, device)
    # generate_digit(model, 1.0, 0.0, device)

    # plot latent space for mean and variance between -1 and 1
    # plot_latent_space(model, device=device)

    # plot latent space for mean and variance between -5 and 5
    # plot_latent_space(model, 5, device=device)

    # close tensorboard writer
    writer.close()


def save_checkpoint(model, epoch, loss, directory='model_checkpoints', filename='checkpoint.pth', optimizer=None):
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


def load_checkpoint(model, directory='model_checkpoints', filename='checkpoint.pth', optimizer=None):
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


def set_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


class Encoder(nn.Module):
    """
    Encodes input to a latent space representation with latent_dim dimensions
    """
    def __init__(self, input_dim=784, latent_dim=200):
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(input_dim, round((input_dim+latent_dim)/2))  # default vals would be 784 > 492
        self.hidden_layer = nn.Linear(round((input_dim+latent_dim)/2), round((input_dim+latent_dim)/4))  # default vals would be 492 > 246

        # latent mean and variance layers
        self.mean_layer = nn.Linear(round((input_dim+latent_dim)/4), latent_dim)  # μ represents center of the Gaussian distribution for each latent variable
        # logvar_layer outputs the log-variance (log(σ^2)) of the Gaussian distribution for each latent variable
        # using log-variance rather than variance directly ensures numerical stability, as variance must be positive.
        # logvar_layer can produce negative values, but they are transformed back into positive variances during reparameterisation.
        self.logvar_layer = nn.Linear(round((input_dim+latent_dim)/4), latent_dim)  # log(σ^2) represents the uncertainty or spread of the Gaussian distribution for each latent variable

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.input_layer(x))
        h2 = self.LeakyReLU(self.hidden_layer(h))

        mean, logvar = self.mean_layer(h2), self.logvar_layer(h2)

        return mean, logvar


class Decoder(nn.Module):
    """
    Decodes a latent vector into a reconstructed output.
    """
    def __init__(self, output_dim=784, latent_dim=200):
        super(Decoder, self).__init__()

        self.hidden_layer = nn.Linear(latent_dim, round((output_dim+latent_dim)/4))  # default vals would be 200 > 246
        self.hidden_layer2 = nn.Linear(round((output_dim+latent_dim)/4), round((output_dim+latent_dim)/2))  # default vals would be 246 > 492
        self.output_layer = nn.Linear(round((output_dim+latent_dim)/2), output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of the Decoder.
        """
        h = self.LeakyReLU(self.hidden_layer(x))
        h2 = self.LeakyReLU(self.hidden_layer2(h))

        recon_x = torch.sigmoid(self.output_layer(h2))  # output values in range [0,1]

        return recon_x


class VAE(nn.Module):
    """
    Integrates both the Encoder and Decoder classes.
    Manages the reparameterisation and latent space operations.
    """
    def __init__(self, input_dim=784, latent_dim=200, device=None):
        super(VAE, self).__init__()
        self.device = device if device is not None else set_device()
        self.latent_dim = latent_dim

        # encoder layer
        self.encoder = Encoder(input_dim, self.latent_dim)

        # decoder layer
        self.decoder = Decoder(input_dim, self.latent_dim)

    def encode(self, x):
        """
        Encodes the input into mean and log-variance of the latent space distribution.
        """
        mean, logvar = self.encoder(x)

        return mean, logvar

    def decode(self, z):
        """
        Decodes the latent vector into a reconstructed output.
        """
        z = self.decoder(z)
        return z

    def reparameterisation(self, mean, logvar):
        """
        Applies the reparameterisation trick to sample from the latent distribution.
        Allows sampling from Gaussian distribution in the latent space in a way that allows gradients to flow through

        :param mean: (torch.Tensor) Mean of the latent space distribution.
        :param logvar: (torch.Tensor) Log-variance of the latent space distribution.
        :return torch.Tensor: Sampled latent vector.
        """
        # epsilon actually reparameterises our VAE network. This allows the mean and log-variance vectors to still remain as the learnable parameters of the network while still maintaining the stochasticity of the entire system
        epsilon = torch.randn_like(logvar).to(self.device)  # sample noise from a normal distribution with mean 0 and variance 1. Values can be negative or positive. Tensor will have same shape as logvar
        std = torch.exp(0.5 * logvar)  # compute standard deviation (σ) and ensures it is always positive
        z = mean + std * epsilon  # reparameterisation trick to allow backpropagation through stochastic sampling

        return z

    def reconstruct(self, mean=0, logvar=1):
        """
        Reconstruct using a single mean and variance value across the entire latent space.
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

    def forward(self, x):
        """
        Forward pass of the VAE model.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterisation(mean, logvar)
        recon_x = self.decode(z)

        return recon_x, mean, logvar


def loss_function(x, recon_x, mean, logvar):
    # pixel values are treated as probabilities, BCE is appropriate for measuring how well the reconstructed image (recon_x) matches the original image (x) on a per-pixel basis
    reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # regularisation loss
    # KL(q(z∣x) ∥ p(z))= −1/2 ∑(1 + log(σ^2) − μ^2 − σ^2) where μ and σ^2 are the mean and variance of the latent distribution q(z∣x), respectively
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # difference between the learned latent distribution (mean and variance) and the prior distribution (typically a standard normal distribution - with mean 0 and variance 1)

    return reconstruction_loss, KLD


def train(train_loader, model, optimizer, device: Optional[torch.device] = set_device()):
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    processed = 0

    for batch_idx, (inputs, _) in enumerate(train_loader):
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


def test(test_loader, model, device: Optional[torch.device] = set_device()):
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    processed = 0

    with torch.no_grad():  # No need for gradients during testing
        for inputs, _ in test_loader:
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


def download_ds_make_dl(batch_size=100):
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


def visualise_dl(train_loader, samples: Optional[int] = 25, figsize: Optional[tuple] = (5, 5)):
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


def generate_digit(model, mean: float, logvar: float, show: bool = True):
    model.eval()
    x_decoded = model.reconstruct(mean, logvar)  # Decode to get the generated digit
    digit = x_decoded.detach().cpu().reshape(28, 28)  # reshape vector to 2d array

    if show is True:
        plt.imshow(digit, cmap='gray')
        plt.axis('off')
        plt.show()

    plt.close()
    return digit


def plot_latent_space(model, mean_range: Union[float, tuple[float, float]] = (-1., 1.), logvar: Union[float, tuple[float, float]] = 1., n: int = 25, digit_size=28, figsize=15, show: bool = True):

    mean_range = format_range(mean_range)
    logvar = format_range(logvar)

    model.eval()

    # define canvas size (where images are placed)
    # n is number of images in the grid, digit_size is size of each digit
    figure = np.zeros((digit_size * n, digit_size * n), dtype=np.float32)  # ensure values in canvas are floats, required for later operations

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
            digit = x_decoded.detach().cpu().reshape(28, 28)  # reshape vector to 2d array
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    # create a figure and an axes object
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(figure, cmap="Greys_r")

    # set title
    plt.title(f'VAE Latent Space Visualisation\nMean Range: {mean_range}, Variance: {logvar}')

    # define where the ticks should appear in relation to images
    start_range = digit_size // 2  # allows first tick mark to align with center of first image in the grid
    end_range = n * digit_size + start_range  # allows last tick mark to align with center of last image in grid
    pixel_range = np.arange(start_range, end_range, digit_size)  # defines where tick marks will be placed along axis

    # Set the tick positions and labels for axes
    ax.set_xticks(pixel_range)
    ax.set_xticklabels(grid_x)
    ax.set_xlabel("Mean")

    ax.set_yticks(pixel_range)
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


def format_range(value: Union[int, float, tuple[Union[int, float], Union[int, float]]]) -> tuple:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Tuple must have exactly 2 elements.")
        return tuple(float(v) for v in value)
    elif isinstance(value, (int, float)):
        return -float(value), float(value)
    else:
        raise ValueError("Input must be an int, float, or a tuple of two ints/floats.")


if __name__ == '__main__':
    main()
