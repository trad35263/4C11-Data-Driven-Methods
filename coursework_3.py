# import modules
import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from time import perf_counter as timer

# import pytorch
import torch
import torch.utils.data as Data
import torch.nn as nn
#import torch.optim as optim

# import CPU monitoring modules
import psutil
import os

# import utils
import utils

# Inputs class
class Inputs:
    """Stores input parameters for the program."""
    # file locations
    data_folder = "../Coursework 3"
    file_name = "viscodata_3mat.mat"
    export_folder = "exports3"

    # number of training and test datapoints
    training_points = 300

    # down sampling factor to reduce temporal resolution
    s = 4

    # hyperparameters
    no_of_layers = 1
    input_size = 1
    hidden_size = 8
    output_size = 1
    no_of_hidden_layers = 1
    channel_width = 4
    dropout = 0

    # learning parameters
    no_of_epochs = 50
    batch_size = 20
    learning_rate = 1e-3
    step_size = 50
    gamma = 0.5
    weight_decay = 1e-4

    # plotting parameters
    figsize = (9, 5)
    fontsize = 12
    titlesize = 14
    dpi = 300

    # terminal output parameters
    print_epoch = 5

# Neural_net class
class Neural_net(nn.Module):
    """Parent class for all neural network architectures."""
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
            for input, target in training_loader:

                # clear gradients
                optimiser.zero_grad()

                # initialise fresh hidden state
                hidden_state = torch.zeros(training_loader.batch_size, self.hidden_size)

                # preallocate tensor of stress predictions
                stress_predicted = torch.zeros(training_loader.batch_size, input.shape[1])

                # apply "warm start" where stress state at t = 0 is given by initial conditions
                stress_predicted[:, 0] = target[:, 0]

                # loop for each time step (starting at t = dt)
                for index in range(1, input.shape[1]):

                    # propagate through RNO to determine stress prediction at next time step
                    stress_predicted[:, index], self.hidden_state = self(
                        input[:, index].unsqueeze(1), input[:, index - 1].unsqueeze(1),
                        hidden_state
                    )

                # calculate loss
                loss = loss_function(stress_predicted, target)

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

                    # initialise fresh hidden state
                    hidden_state = torch.zeros(test_loader.batch_size, self.hidden_size)

                    # preallocate tensor of stress predictions
                    stress_predicted = torch.zeros(test_loader.batch_size, input.shape[1])

                    # apply "warm start" where stress state at t = 0 is given by initial conditions
                    stress_predicted[:, 0] = target[:, 0]

                    # loop for each time step (starting at t = dt)
                    for index in range(1, input.shape[1]):

                        # propagate through RNO to determine stress prediction at next time step
                        stress_predicted[:, index], self.hidden_state = self(
                            input[:, index].unsqueeze(1), input[:, index - 1].unsqueeze(1),
                            hidden_state
                        )

                    # calculate loss
                    loss = loss_function(stress_predicted, target)

                    # update test loss
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

        # set y-axis ticks IMPORTANT
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5, subs=[1, 2, 5]))

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

# RNO class
class RNO(Neural_net):
    """Stores the parameters for a Recurrent Neural Operator (RNO)."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, no_of_hidden_layers, channel_width, dropout, dt):
        """Creates an instance of the RNO class."""
        # instantiate parent class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.no_of_hidden_layers = no_of_hidden_layers
        self.channel_width = channel_width
        self.dropout = dropout
        self.dt = dt

        # define lists of layer sizes
        layer_sizes = (
            [input_size + output_size + hidden_size]
            + [channel_width] * (no_of_layers - 1)
            + [output_size]
        )
        hidden_layer_sizes = (
            [output_size + hidden_size]
            + [channel_width] * (no_of_hidden_layers - 1)
            + [hidden_size]
        )

        # create module list for main feedforward network and loop over layer sizes
        self.layers = nn.ModuleList()
        for index in range(no_of_layers):

            # append layer with corresponding input and output sizes
            self.layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))

            # for all but final layer
            if index != no_of_layers - 1:

                # append non-linear activation function
                self.layers.append(nn.SELU())

        # create module list for hidden feedforward network and loop over (hidden) layer sizes
        self.hidden_layers = nn.ModuleList()
        for index in range(no_of_hidden_layers):

            # append layer with corresponding input and output sizes
            self.hidden_layers.append(nn.Linear(hidden_layer_sizes[index], hidden_layer_sizes[index + 1]))

            # for all but final (hidden) layer
            if index != no_of_hidden_layers - 1:

                # append non-linear activation function
                self.hidden_layers.append(nn.SELU())

        # give label for plotting
        self.label = "RNO"
        self.quantities = [
            ["no_of_hidden_layers", "No. of Hidden Layers"],
            ["channel_width", "Channel Width"],
            ["dropout", "Dropout"]
        ]

    def forward(self, current_strain, previous_strain, hidden_state):
        """Propagates forwards through the neural network and returns the output."""
        # get current rate of change of hidden state
        dh_dt = torch.cat((previous_strain, hidden_state), 1)

        # loop for each hidden layer
        for hidden_layer in self.hidden_layers:

            # update rate of change of hidden state
            dh_dt = hidden_layer(dh_dt)

        # update hidden state via Euler integration
        hidden_state = dh_dt * self.dt + hidden_state

        # get current state of network
        x = torch.cat((previous_strain, (previous_strain - current_strain) / self.dt, hidden_state), 1)

        # loop for each layer
        for layer in self.layers:

            # pass intermediate tensor through layer
            x = layer(x)

        # return output
        output = x.squeeze(1)
        return output, hidden_state

    def initHidden(self,b_size):
        """Returns a tensor of zeros with size equal to (batch_size, hidden_size)."""
        return torch.zeros(b_size, self.hidden_size)
    
# DenseNet class
class DenseNet(nn.Module):
    """Stores parameters relevant to the RNO model."""
    def __init__(self, layers, nonlinearity):
        """Creates an instance of the DenseNet class."""
        # instantiate parent class
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        """Creates an instance of the MatReader class."""
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# Normaliser class
class Normaliser():
    """Stores the parameters for a Z-score normalisation encoder and decoder."""
    def __init__(self, x, eps = 1e-6):
        """Creates an instance of the Normaliser class."""
        # store input variables
        self.eps = eps

        # calculate mean and standard deviation of data
        self.mean = torch.mean(x)
        self.std = torch.std(x)

    def encode(self, x):
        """Returns the normalised value of an input."""
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        """Reverses the normalisation and returns the original input."""
        x = x * (self.std + self.eps) + self.mean
        return x

# save_figure function
def save_figure(fig, ax, filename):

    # save plot
    fig.savefig(f"{Inputs.export_folder}/{filename}", dpi = Inputs.dpi)

# main function
def main():

    # load data
    data_loader = MatReader(Inputs.data_folder + "/" + Inputs.file_name)

    # get input and output data
    n_total = data_loader.data["epsi_tol"].shape[0]
    data_input = data_loader.read_field("epsi_tol").contiguous().view(n_total, -1)
    data_output = data_loader.read_field("sigma_tol").contiguous().view(n_total, -1)

    # down sample the data to reduce temporal resolution
    data_input = data_input[:, 0::Inputs.s]
    data_output = data_output[:, 0::Inputs.s]

    # normalise data
    input_normaliser = Normaliser(data_input)
    output_normaliser = Normaliser(data_output)
    data_input = input_normaliser.encode(data_input)
    data_output = output_normaliser.encode(data_output)

    # create training data loader
    training_set = Data.TensorDataset(data_input[:Inputs.training_points, :], data_output[:Inputs.training_points, :])
    training_loader = Data.DataLoader(training_set, Inputs.batch_size, shuffle = True)

    # create test data loader with batch size equal to entire test dataset
    test_set = Data.TensorDataset(data_input[Inputs.training_points:, :], data_output[Inputs.training_points:, :])
    test_loader = Data.DataLoader(test_set, len(test_set), shuffle = False)

    # define time increment to use
    dt = 1 / (data_output.shape[1] - 1)

    # define loss function
    loss_function = nn.MSELoss()

    # create neural network instance
    net = RNO(
        Inputs.no_of_layers, Inputs.input_size, Inputs.hidden_size, Inputs.output_size,
        Inputs.no_of_hidden_layers, Inputs.channel_width, Inputs.dropout, dt
    )

    # print number of parameters
    net.print_params()

    # define optimiser and scheduler
    optimiser = torch.optim.Adam(net.parameters(), lr = Inputs.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size = Inputs.step_size, gamma = Inputs.gamma
    )

    # train neural network
    net.training_loop(
        Inputs.no_of_epochs, training_loader, test_loader, loss_function, optimiser, scheduler
    )

    # plot convergence history
    net.plot_convergence()

# upon script execution
if __name__ == "__main__":

    # run main()
    main()

    # show all plots
    plt.show()
