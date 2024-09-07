import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from typing import Optional

# Set the backend (Option 1 or 2)
matplotlib.use('TkAgg')  # or 'Agg' if you're in a headless environment


def main():
    # define hyperparameters
    learning_rate = 1e-3
    batch_size = 100
    epochs = 3  # 50

    train_ds, test_ds, train_loader, test_loader = download_ds_make_dl(batch_size)

    device = set_device()

    visualise_dl(train_loader)

    model = VAE(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        average_loss = train(train_loader, model, optimizer, device)
        print(f"Epoch {t + 1} average loss: {average_loss:>8f}")
        average_loss = test(test_loader, model, device)
        print(f"Test set average loss: {average_loss:>8f}")

        # save model if loss is least so far ----------------------------------------
    print("Done!")

    # generate a single digit using the decoder passing in mean and variance values
    generate_digit(model, 0.0, 1.0, device)
    generate_digit(model, 1.0, 0.0, device)

    # plot latent space for mean and variance between -1 and 1
    plot_latent_space(model, device=device)

    # plot latent space for mean and variance between -5 and 5
    plot_latent_space(model, 5, device=device)


def set_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


class Encoder(nn.Module):
    """
    Encodes input to a latent space representation with latent_dim dimensions
    Outputs the latent representation directly without the mean and log_var layers
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        return x


class Decoder(nn.Module):
    """
    Decodes a latent vector into a reconstructed output.

    The latent vector is transformed by the VAE class from the latent space (mean and variance) into the latent dimension.

    Args:
        output_dim (int): The dimension of the reconstructed output (e.g., 784 for a 28x28 image).
        hidden_dim (int): The dimension of the hidden layer in the Decoder.
        latent_dim (int): The dimension of the latent space (should match the dimension of the latent vector input).

    Forward Pass:
        - Takes a latent vector `x` as input, which has been transformed to the latent dimension.
        - Passes this vector through a series of activation functions and a final output layer.
        - Outputs a reconstructed image of the original data dimension, typically using a sigmoid activation function
          to ensure pixel values are between 0 and 1.

    The Decoder learns to map this latent vector back to the data space to reconstruct the original data.
    """
    def __init__(self, output_dim=784, hidden_dim=400, latent_dim=200):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of the Decoder.

        :param x: (torch.Tensor) Latent vector of shape (batch_size, latent_dim), which has been transformed to
                              latent space dimension by the VAE class.
        :return torch.Tensor: Reconstructed output of shape (batch_size, output_dim).
        """
        x = self.LeakyReLU(x)
        x = self.LeakyReLU(self.linear1(x))

        x_hat = torch.sigmoid(self.output(x))

        return x_hat


class VAE(nn.Module):
    """
    Integrates both the Encoder and Decoder classes.
    Manages the reparameterisation and latent space operations.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=None):
        super(VAE, self).__init__()
        self.device = device if device is not None else set_device()

        # encoder layer
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)

        # latent mean and variance layers
        self.mean_layer = nn.Linear(latent_dim, 2)  # μ represents center of the Gaussian distribution for each latent variable
        # logvar_layer outputs the log-variance (log(σ^2)) of the Gaussian distribution for each latent variable
        # using log-variance rather than variance directly ensures numerical stability, as variance must be positive.
        # logvar_layer can produce negative values, but they are transformed back into positive variances during reparameterisation.
        self.logvar_layer = nn.Linear(latent_dim, 2)  # log(σ^2) represents the uncertainty or spread of the Gaussian distribution for each latent variable

        # latent space to latent dimension transformation
        self.latent_transform = nn.Linear(2, latent_dim)  # transform from 2D latent space (mean and variance) to latent space dimension

        # decoder layer
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def encode(self, x):
        """
        Encodes the input into mean and log-variance of the latent space distribution.

        :param x: (torch.Tensor) Input data.
        :return tuple: (mean, logvar) where mean and logvar are the parameters of the latent space distribution.
        """
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)

        return mean, logvar

    def reparameterisation(self, mean, log_var):
        """
        Applies the reparameterisation trick to sample from the latent distribution.
        Allows sampling from Gaussian distribution in the latent space in a way that allows gradients to flow through

        :param mean: (torch.Tensor) Mean of the latent space distribution.
        :param log_var: (torch.Tensor) Log-variance of the latent space distribution.
        :return torch.Tensor: Sampled latent vector.
        """
        epsilon = torch.randn_like(log_var).to(self.device)  # sample noise from standard normal
        std = torch.exp(0.5 * log_var)  # compute standard deviation (σ) and ensures it is always positive
        z = mean + std * epsilon  # reparameterisation trick to allow backpropagation through stochastic sampling

        return z

    def decode(self, z):
        """
        Decodes the latent vector into a reconstructed output.

        :param z: (torch.Tensor) Latent vector sampled from the latent space distribution.
        :return torch.Tensor: Reconstructed output from the decoder.
        """
        z_latent = self.latent_transform(z)  # Transform the latent vector to the latent space dimension
        return self.decoder(z_latent)

    def forward(self, x):
        """
        Forward pass of the VAE model.

        :param x: (torch.Tensor) Input data.
        :return tuple: (x_hat, mean, logvar) where x_hat is the reconstructed output, and mean and logvar are the parameters of the latent space distribution.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterisation(mean, logvar)
        x_hat = self.decode(z)

        return x_hat, mean, logvar


def loss_function(x, x_hat, mean, log_var):
    # pixel values are treated as probabilities, BCE is appropriate for measuring how well the reconstructed image (x_hat) matches the original image (x) on a per-pixel basis
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL(q(z∣x) ∥ p(z))= −1/2 ∑(1 + log(σ^2) − μ^2 − σ^2) where μ and σ^2 are the mean and variance of the latent distribution q(z∣x), respectively
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # difference between the learned latent distribution (mean and variance) and the prior distribution (typically a standard normal distribution - with mean 0 and variance 1)

    return reconstruction_loss + KLD


def train(train_loader, model, optimizer, device: Optional[torch.device] = set_device()):
    model.train()

    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, _) in enumerate(train_loader):
        # flatten the inputs based on their shape (from 28x28 to a vector of size 784 (MNIST))
        batch_size = inputs.size(0)
        num_features = torch.prod(torch.tensor(inputs.shape[1:])).item()

        inputs_flattened = inputs.view(batch_size, num_features).to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(inputs_flattened)
        loss = loss_function(inputs_flattened, x_hat, mean, log_var)

        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        optimizer.step()

    average_loss = total_loss / num_batches

    return average_loss


def test(test_loader, model, device: Optional[torch.device] = set_device()):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # No need for gradients during testing
        for inputs, _ in test_loader:
            batch_size = inputs.size(0)
            num_features = torch.prod(torch.tensor(inputs.shape[1:])).item()
            inputs_flattened = inputs.view(batch_size, num_features).to(device)

            x_hat, mean, log_var = model(inputs_flattened)
            loss = loss_function(inputs_flattened, x_hat, mean, log_var)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)

    return average_loss


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


def generate_digit(model, mean: float, var: float, device: Optional[torch.device] = set_device()):
    model.eval()
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)  # Latent vector (mean and variance)
    # z_sample = torch.randn(1, 2).to(device)  # Sample a latent vector from standard normal distribution
    x_decoded = model.decode(z_sample)  # Decode to get the generated digit
    digit = x_decoded.detach().cpu().reshape(28, 28)  # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_latent_space(model, scale=1.0, n=25, digit_size=28, figsize=15, device: Optional[torch.device] = set_device()):
    model.eval()

    # display an n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


if __name__ == '__main__':
    main()
