# import modules
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import perf_counter as timer
import h5py

# import modules for debugging
import psutil
import os

# import utils module
import utils

# Inputs class
class Inputs:
    """Stores input parameters for the program."""
    # file paths for training/test data and saving plots
    data_folder = "../Coursework 2/Coursework2/Coursework2_Problem_2/"
    training_path = "Darcy_2D_data_train.mat"
    test_path = "Darcy_2D_data_test.mat"
    export_folder = "exports"

    # network hyperparameters
    input_size = None
    output_size = None      # input/output size is a 3D tensor than can be inferred from data
    no_of_layers = 2
    channel_width = 16
    dropout = 0
    modes = 4

    # training variables
    no_of_epochs = 5
    batch_size = 20
    learning_rate = 1e-3
    weight_decay = 1e-4
    step_size = 50
    gamma = 0.5

    # plotting parameters
    figsize = (10, 6)
    fontsize = 12
    titlesize = 14
    dpi = 300

    # terminal output parameters
    print_epoch = 5

    # seed for plotting RNG
    plot_seed = 42

# Lploss class
class LpLoss(object):
    """"""
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """"""
        #
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # print('x.shape',x.shape)
        # print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)
    
# UnitGaussianNormaliser class
class UnitGaussianNormaliser(object):
    """Stores the parameters for a pointwise Gaussian normalisation encoder and decoder."""
    def __init__(self, x, eps = 1e-5):
        """Creates an instance of the UnitGaussianNormaliser class."""
        # initialise instance of parent class
        super(UnitGaussianNormaliser, self).__init__()

        # store input variables
        self.eps = eps

        # calculate mean and standard deviation
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

# Neural_net class
class Neural_net(nn.Module):
    """Parent class for all avaiable neural network options."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size):
        # initialise instance of nn.Module class
        super().__init__()

        # store input variables
        self.no_of_layers = no_of_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # create empty lists to store training loss
        self.training_loss = []
        self.test_loss = []
        self.time = []

    def forward(self, x):
        """Propagates forwards through the neural network architecture."""
        # calculate the neural network output
        return self.net(x)

    def print_params(self):
        """Prints the number of parameters stored in the neural net."""
        # calculate number of parameters and print
        self.no_of_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {utils.Colours.GREEN}{self.no_of_params}{utils.Colours.END}")

    def training_loop(
            self, no_of_epochs, training_loader, test_loader, loss_function, optimiser, scheduler
        ):
        """Trains the neural network."""
        # store input variables
        self.no_of_epochs = no_of_epochs

        # print number of epochs
        print(
            f"Start training {utils.Colours.GREEN}{self.label}{utils.Colours.END} for "
            f"{utils.Colours.GREEN}{no_of_epochs}{utils.Colours.END} epochs...")
        t_start = timer()

        # loop for each epoch
        for epoch in range(no_of_epochs):

            # set neural net to train and training loss to zero
            self.train(True)
            training_loss = 0

            # loop for each entry in the training data loader
            for data in training_loader:

                # propagate forward through the neural network and calculate loss
                input, target = data
                output = self(input)
                loss = loss_function(output, target)

                # clear gradients
                optimiser.zero_grad()

                # calculate derivatives of loss with respect to learnt parameters
                loss.backward()
                
                # update parameters
                optimiser.step()

                # sum training loss
                training_loss += loss.item()

            # step scheduler
            scheduler.step()

            # enter evaluation mode and disable gradient tracking
            self.eval()
            with torch.no_grad():

                # initialise test loss as zero
                test_loss = 0

                # loop through test data
                for input, target in test_loader:

                    # define forward neural network
                    output = self(input)

                    # calculate loss
                    loss = loss_function.forward(output, target)

                    # update training loss
                    test_loss += loss.item()

            # end timer
            t = timer()

            # every Nth epoch
            if (epoch + 1) % Inputs.print_epoch == 0 or epoch == 0:

                # print user feedback
                print(
                    f"Epoch: {utils.Colours.GREEN}{epoch + 1}{utils.Colours.END}, "
                    f"Training loss: {utils.Colours.GREEN}{training_loss / len(training_loader):.4g}"
                    f"{utils.Colours.END}, "
                    f"Test loss: {utils.Colours.GREEN}{test_loss / len(test_loader):.4g}{utils.Colours.END}"
                )

                # calculate and print memory usage to debug memory leak
                process = psutil.Process(os.getpid())
                print(
                    f"Memory usage: {utils.Colours.GREEN}{process.memory_info().rss / 1e6:.1f}"
                    f"{utils.Colours.END} MB, "
                    f"Time taken: {utils.Colours.GREEN}{t - t_start:.4g}{utils.Colours.END} s, "
                    f"Time remaining: {utils.Colours.GREEN}{(t - t_start) * (self.no_of_epochs / (epoch + 1) - 1):.4g}{utils.Colours.END} s, "
                )

            # append training and test loss
            self.training_loss.append(training_loss / len(training_loader))
            self.test_loss.append(test_loss / len(test_loader))
            self.time.append(t - t_start)

    def plot_convergence(self, ymin = None, ymax = None):
        """Plots the convergence history of the training and testing."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)
        
        # plot training and test loss against epoch
        ax.plot(np.arange(self.no_of_epochs), self.training_loss, label = "Training")
        ax.plot(np.arange(self.no_of_epochs), self.test_loss, label = "Test")

        # configure primary axis
        ax.grid()
        ax.set_xlabel("No. of Epochs", fontsize = Inputs.fontsize)
        ax.set_ylabel("Loss", fontsize = Inputs.fontsize)
        ax.set_yscale("log")

        # axis bounds are given as input arguments
        if ymin is not None and ymax is not None:

            # set y-axis bounds
            ax.set_ylim(ymin, ymax)

        # create twin y axis for time
        ax_time = ax.twinx()
        ax_time.plot(
            np.arange(self.no_of_epochs), self.time,
            color = 'grey', linestyle = '--', alpha = 0.5, label = "Time"
        )
        ax_time.set_ylabel("Time (s)", fontsize = Inputs.fontsize)
        ax_time.tick_params(axis = 'y')

        # combine legends from both axes
        lines_loss, labels_loss = ax.get_legend_handles_labels()
        lines_time, labels_time = ax_time.get_legend_handles_labels()
        ax.legend(
            lines_loss + lines_time, labels_loss + labels_time, loc = "center left",
            bbox_to_anchor = (1.1, 0.5), fontsize = Inputs.fontsize
        )

        # determine title text
        title_text = f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params}"
        for quantity in self.quantities:

            title_text += f" | {quantity[1]}: {getattr(self, quantity[0])}"
        
        # set title
        ax.text(
            0.5, 1.03, title_text,
            transform = ax.transAxes, ha = 'center', va = 'bottom',
            fontsize = Inputs.titlesize
        )

        # tight layout
        plt.tight_layout()

        # construct filename
        filename = (
            f"plot_convergence_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, ax, filename)

    def plot_predictions(
            self, a_test, u_test, a_normaliser, u_normaliser,
            no_of_samples = 3, ymin = None, ymax = None
        ):
        """Plot input, ground truth, prediction and error for n_samples test cases."""
        # generate random sample indices and reorder
        rng = np.random.default_rng(Inputs.plot_seed)
        indices = rng.choice(len(a_test), size = no_of_samples, replace = False)
        indices.sort()

        # set neural net to test mode and deactivate gradient tracking
        self.eval()
        with torch.no_grad():

            # calculate values predicted by the neural net
            u_predicted = self(a_test[indices])

        # decode values and convert to numpy for plotting
        a_plot = a_normaliser.decode(a_test[indices]).numpy()
        u_true = u_normaliser.decode(u_test[indices]).numpy()
        u_predicted = u_normaliser.decode(u_predicted).numpy()

        # calculate error
        u_error = np.abs(u_true - u_predicted)

        # create plot of subplots
        fig, axes = plt.subplots(no_of_samples, 4, figsize = Inputs.figsize)

        # loop for each plot column
        headings = ["Input", "Ground Truth", "Prediction", "Error"]
        for column, heading in enumerate(headings):

            # set column heading text
            axes[0, column].text(
                0.5, 1.08, heading, transform = axes[0, column].transAxes, ha = 'center',
                va = 'bottom', fontsize = Inputs.fontsize, fontweight = 'bold'
            )

        # calculate colour bar ranges
        vmin = np.zeros(3)
        vmax = np.zeros(3)
        vmin[0] = np.min(a_plot)
        vmax[0] = np.max(a_plot)
        vmin[1] = np.min(u_true)
        vmax[1] = np.max(u_true)

        # error plot bounds are given as input arguments
        if ymin is not None and ymax is not None:

            # set as colour bar range
            vmin[2] = ymin
            vmax[2] = ymax

        else:
            
            vmin[2] = np.min(u_error)
            vmax[2] = np.max(u_error)

        # loop for each sample
        for index in range(no_of_samples):

            # row label on the left of each row
            axes[index, 0].text(
                -0.25, 0.5, f"Sample {indices[index]}", transform = axes[index, 0].transAxes,
                ha = 'right', va = 'center', fontsize = Inputs.fontsize, fontweight = 'bold',
                rotation = 90
            )

            # plot input "a"-values with its own colour scale and colour map
            im = axes[index, 0].imshow(
                a_plot[index], vmin = vmin[0], vmax = vmax[0], cmap = "plasma",
                origin = "lower"
            )
            fig.colorbar(im, ax = axes[index, 0])

            # plot ground truth "u"-values
            im = axes[index, 1].imshow(
                u_true[index], vmin = vmin[1], vmax = vmax[1], cmap = "viridis",
                origin = "lower"
            )
            fig.colorbar(im, ax = axes[index, 1])

            # plot "u"-values as predicted by neural network
            im = axes[index, 2].imshow(
                u_predicted[index], vmin = vmin[1], vmax = vmax[1], cmap = "viridis",
                origin = "lower"
            )
            fig.colorbar(im, ax = axes[index, 2])

            # plot error between ground truth and neural net prediction
            im = axes[index, 3].imshow(
                u_error[index], vmin = vmin[2], vmax = vmax[2], cmap = "Reds",
                origin = "lower"
            )
            fig.colorbar(im, ax = axes[index, 3])

            # loop for each column
            for column, heading in enumerate(headings):

                # remove x- and y-axis ticks
                axes[index, column].set_xticks([])
                axes[index, column].set_yticks([])

        # determine title text
        title_text = (
            f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params} | "
            f"Epochs: {self.no_of_epochs}"
        )
        for quantity in self.quantities:

            title_text += f" | {quantity[1]}: {getattr(self, quantity[0])}"

        # set title
        fig.suptitle(title_text, fontsize = Inputs.titlesize)

        # tight layout
        plt.tight_layout()

        # construct filename
        filename = (
            f"plot_predictions_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, axes, filename)

# CNN class
class CNN(Neural_net):
    """Convolutional Neural Network (CNN) model."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, dropout):
        """Creates an instance of the CNN class."""
        # instantiate parent class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.dropout = dropout

        # channel width is equivalent to hidden size for FCNN
        self.channel_width = hidden_size

        # calculate number of encoder layers, ensuring total number of layers is even
        self.no_of_layers = 2 * no_of_layers // 2
        no_of_encoder_layers = int(self.no_of_layers / 2)

        # build encoder layers
        encoder = []
        for index in range(no_of_encoder_layers):
            dilation = 2**index
            in_channels = 1 if index == 0 else self.channel_width * (2**(index - 1))
            out_channels = self.channel_width * (2**index)
            encoder += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size = 3, padding = dilation,
                    dilation = dilation
                ),
                nn.ReLU()
            ]

        # build decoder layers
        decoder = []
        for index in range(no_of_encoder_layers - 1, -1, -1):
            dilation = 2 ** index
            in_channels = self.channel_width * (2 ** index)
            out_channels = self.channel_width * (2 ** (index - 1)) if index > 0 else 1
            decoder += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size = 3, padding = dilation,
                    dilation = dilation
                ),
                nn.ReLU() if index > 0 else nn.Identity()
            ]

        # combine encoder and decoder into sequential
        self.layers = nn.Sequential(*encoder, *decoder)

        # give label for plotting
        self.label = "CNN"
        
        # specify variables which should be appended to plot titles
        self.quantities = [["dropout", "Dropout"]]

    def forward(self, x):
        """Propagates through the neural network and returns the output."""
        # pass x through neural net and return output
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.squeeze(1)
        return out

# SpectralConv2d class
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# MLP class
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x

# FNO class
class FNO(Neural_net):
    """Fourier Neural Operator (FNO) model."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, modes):
        """Creates an instance of the FNO class."""
        # instantiate parent class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.modes = modes
        self.modes1 = modes
        self.modes2 = modes
        self.width = hidden_size
        self.dropout = 0

        # lifting layer: maps (a(x,y), x, y) -> width channels
        self.p = nn.Linear(3, self.width)

        # spectral convolution layers
        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(no_of_layers)
        ])

        # pointwise MLPs
        self.mlps = nn.ModuleList([
            MLP(self.width, self.width, self.width)
            for _ in range(no_of_layers)
        ])

        # local bypass convolutions
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(no_of_layers)
        ])

        # activations
        self.acts = nn.ModuleList([
            nn.GELU()
            for _ in range(no_of_layers)
        ])

        # projection layer: maps width channels -> 1 output channel
        self.q = MLP(self.width, 1, self.width * 4)

        # set label for plotting
        self.label = "FNO"
        
        # specify variables which should be appended to plot titles
        self.quantities = [["modes", "Modes"]]

    def forward(self, x):
        grid = self.get_grid(x.shape)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        # loop through FNO layers
        for i in range(self.no_of_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            # apply activation to all layers except the last
            if i < self.no_of_layers - 1:
                x = self.acts[i](x)

        x = self.q(x)
        x = x.squeeze(1)
        return x

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

# plot_nets function
def plot_nets(nets):
    """Creates a scatter plot of the different neural network performances."""
    # create plot
    fig, ax = plt.subplots()

    # scatter training data against number of parameters
    ax.scatter(
        [net.no_of_params for net in nets],
        [net.test_loss[-1] for net in nets]
    )

    ax.set_xscale('log')  # parameters vary over orders of magnitude
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Final test loss')
    ax.set_title("Neural Networks Comparison")

    plt.tight_layout()

    save_figure(fig, ax, "plot_nets")

# save_figure function
def save_figure(fig, ax, filename):

    # save plot
    fig.savefig(f"{Inputs.export_folder}/{filename}", dpi = Inputs.dpi)

# main function
def main():

    # read training data from .mat file
    data_reader = MatRead(Inputs.data_folder + Inputs.training_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    # read test data from .mat file
    data_reader = MatRead(Inputs.data_folder + Inputs.test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # create normaliser objects based on training data for a and u
    a_normaliser = UnitGaussianNormaliser(a_train)
    u_normaliser = UnitGaussianNormaliser(u_train)

    # normalise training data
    a_train = a_normaliser.encode(a_train)
    u_train = u_normaliser.encode(u_train)

    # normalise test data
    a_test = a_normaliser.encode(a_test)
    u_test = u_normaliser.encode(u_test)

    # create training data loader
    training_set = Data.TensorDataset(a_train, u_train)
    training_loader = Data.DataLoader(training_set, Inputs.batch_size, shuffle = True)

    # create test data loader with batch size equal to entire test dataset
    test_set = Data.TensorDataset(a_test, u_test)
    test_loader = Data.DataLoader(test_set, len(test_set), shuffle = False)

    # list of neural networks to investigate
    nets = [
        # example usage:
        #CNN(
        #    Inputs.no_of_layers, Inputs.input_size, Inputs.channel_width, Inputs.output_size,
        #    Inputs.dropout
        #),
        CNN(2, None, 8, None, 0),
        CNN(2, None, 16, None, 0),
        CNN(4, None, 16, None, 0),
        CNN(4, None, 16, None, 0.1),
        # example usage:
        #FNO(
        #    Inputs.no_of_layers, Inputs.input_size, Inputs.channel_width, Inputs.output_size,
        #    modes = Inputs.modes
        #)
        FNO(2, None, 8, None, 1),
        FNO(2, None, 8, None, 2),
        FNO(2, None, 8, None, 4),
        FNO(2, None, 8, None, 8),
        FNO(4, None, 8, None, 4),
        FNO(6, None, 8, None, 4),
        FNO(6, None, 16, None, 4)
    ]

    # loop for each network
    for net in nets:

        # calculate and print number of parameters
        net.print_params()

        # define loss function
        loss_function = LpLoss()

        # define optimiser
        optimiser = torch.optim.AdamW(
            net.parameters(), lr = Inputs.learning_rate,
            weight_decay = Inputs.weight_decay, foreach = False
        )

        # define scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size = Inputs.step_size, gamma = Inputs.gamma
        )

        # train neural network
        net.training_loop(
            Inputs.no_of_epochs, training_loader, test_loader, loss_function, optimiser, scheduler
        )

        # set neural net to test mode and deactivate gradient tracking
        """net.eval()
        with torch.no_grad():

            # calculate values predicted by the neural net
            u_predicted = net(a_test)

        # decode values and convert to numpy for plotting
        u_true = u_normaliser.decode(u_test).numpy()
        u_predicted = u_normaliser.decode(u_predicted).numpy()

        # calculate error between ground truth and predicted
        net.u_error = np.abs(u_true - u_predicted)"""

    # determine axis limits for convergence plot
    ymin = np.min([[net.test_loss, net.training_loss] for net in nets])
    ymax = np.max([[net.test_loss, net.training_loss] for net in nets])

    print(f"ymin: {ymin}")
    print(f"ymax: {ymax}")

    # loop for each net
    for net in nets:

        # produce convergence and prediction plots
        net.plot_convergence(ymin, ymax)
        net.plot_predictions(a_test, u_test, a_normaliser, u_normaliser)

# upon script execution
if __name__ == '__main__':

    # run main
    main()

    # show all plots
    #plt.show()
