# import modules
import matplotlib.pyplot as plt
import numpy as np
import h5py
from time import perf_counter as timer
import random

# import pytorch
import torch
import torch.nn as nn
import torch.utils.data as Data

# import neural network classes
from neural_nets import FCNN, ResNet, U_Net

# import colours
from utils import Colours as colours

# create Inputs class
class Inputs:
    """Stores high-level inputs for the program and resultant neural net."""
    # file path of folder contain material data
    data_folder = "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/4C11 Data-Driven Methods/Project_1/Problem_2_student/Data"
    file_name = "/Eiffel_data.mat"

    # batch size
    batch_size = 32

    # number of training datapoints
    ntrain = 800

    # number of training epochs
    no_of_epochs = 500

    # global figure size and fontsize
    figsize = (7, 5)
    fontsize = 12
    fontsize_title = 14

    # global neural network parameters
    input_size = 20
    hidden_size = 128
    output_size = 1
    no_of_layers = 4

    # neural network parameters (specific to certain types)
    dropout = 0.5

# create Outputs class
class Outputs(Inputs):
    """Stores the most important outputs of the program."""
    def __init__(
            self,
            neural_net, input, output, t_start
        ):
        """Creates an instance of the Outputs class."""
        # store input variables
        self.neural_net = neural_net
        self.input = input
        self.output = output
        self.t_start = t_start

        # print time taken
        self.t_end = timer()
        print(f"Completed after {colours.GREEN}{self.t_end - self.t_start:.4g}{colours.END} s!")

        # print final losses
        print(f"Final training loss {colours.GREEN}{self.neural_net.training_loss[-1]}{colours.END}")
        print(f"Final test loss {colours.GREEN}{self.neural_net.test_loss[-1]}{colours.END}")

class Loss_function(nn.Module):
    """Binary cross-entropy loss with logits."""
    def __init__(self):
        """Creates an instance of the LossFunc class."""
        #
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        """"""
        #
        return self.bce(prediction, target.float())

# create Mat_reader class
class Mat_reader(object):
    """Reads the matlab data from the .mat file provided."""
    def __init__(self, file_path):
        """Creates an instance of the Mat_reader class."""
        # initialise input object if necessary
        super().__init__()

        # store input file path
        self.file_path = file_path

        # read data
        self.data = h5py.File(self.file_path)

    def get_data(self, key):
        """Reads a given set of data from the data object."""
        #
        data = np.array(self.data[key]).transpose()
        return torch.tensor(data, dtype = torch.float32)

# Dataset class
class Dataset():
    """Normalises tensors."""
    def __init__(self, data):
        """Creates an instance of the Dataset class."""
        # store input variables
        self.data = data

        # specify the training and test data
        self.training_data = self.data[:Inputs.ntrain]
        self.test_data = self.data[len(self.data) - Inputs.ntrain:]

        # calculate mean and standard deviation based on training data only
        self.mean = self.training_data.mean(dim = (0, 1), keepdim = True)
        self.std = self.training_data.std(
            dim = (0, 1), unbiased = False, keepdim = True
        )

        # normalise data
        self.training_data_normalised = self.normalise(self.training_data)
        self.test_data_normalised = self.normalise(self.test_data)

    def normalise(self, x):
        """Normalises a given data using the mean and standard deviation."""
        # calculate normalised values
        return (x - self.mean) / self.std

    def inverse_normalise(self, x):
        """Reverses the normalisation process for a given data."""
        # calculate unnormalised value(s)
        return x * self.std + self.mean

# main function
def main():
    """Runs on script execution."""
    # set up data reader
    path = Inputs.data_folder + Inputs.file_name
    data_reader = Mat_reader(path)

    # read data
    loading_data = data_reader.get_data("load_apply")
    result_data = data_reader.get_data("result")

    # shuffle data pseudo-randomly
    rng = np.random.default_rng(seed = 42)
    shuffled_indices = rng.permutation(len(loading_data))
    loading_data = loading_data[shuffled_indices]
    result_data = result_data[shuffled_indices]

    # store data in Dataset classes
    loading = Dataset(loading_data)
    result = Dataset(result_data)

    # print user feedback
    print(f"Number of training samples: {colours.GREEN}{Inputs.ntrain}{colours.END}")
    print(
        f"Number of test samples: {colours.GREEN}{len(result.data) - Inputs.ntrain}{colours.END}"
    )

    # create data loader for training data
    train_set = Data.TensorDataset(loading.training_data_normalised, result.training_data)
    train_loader = Data.DataLoader(train_set, Inputs.batch_size, shuffle = True)

    # create data loader for test data
    test_set  = Data.TensorDataset(loading.test_data_normalised, result.test_data)
    test_loader = Data.DataLoader(test_set, Inputs.batch_size, shuffle = False)

    # define loss function
    loss_function = Loss_function()

    # create neural networks
    neural_nets = [
        FCNN(
            Inputs.no_of_layers, Inputs.input_size, Inputs.hidden_size, Inputs.output_size
        ),
        ResNet(
            Inputs.no_of_layers, Inputs.input_size, Inputs.hidden_size, Inputs.output_size,
            dropout = Inputs.dropout
        ),
        U_Net(
            Inputs.no_of_layers, Inputs.input_size, Inputs.hidden_size, Inputs.output_size,
            dropout = Inputs.dropout
        )
    ]

    # loop for each neural net
    for neural_net in neural_nets:

        # define optimiser
        optimiser = torch.optim.Adam(
            neural_net.parameters(),
            lr = 1e-3,
            weight_decay = 0.0   # set >0 for L2 regularisation
        )

        # define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode = "min",
            factor = 0.5,
            patience = 10
        )

        # train neural network
        neural_net.training_loop(
            Inputs.no_of_epochs, train_loader, test_loader, loss_function, optimiser, scheduler
        )

        # plot convergence history
        neural_net.plot_convergence(Inputs.figsize, Inputs.fontsize, Inputs.fontsize_title)

        # calculate accuracy
        neural_net.calculate_accuracy(test_loader)
        neural_net.error_percent = 100 * (1 - neural_net.accuracy)

    # create bar chart
    plot_attribute(neural_nets, "no_of_params", "No. of Parameters", "Parameter Count Comparison")
    plot_attribute(neural_nets, "error_percent", "Test Error Rate (%)", "Error Rate Comparison")

def plot_attribute(neural_nets, attribute, ylabel, title):
    """Plots the number of parameters of the neural nets in a bar chart."""
    # create plot
    fig, ax = plt.subplots(figsize = Inputs.figsize)
    
    # store parameters counts and labels
    values = [getattr(neural_net, attribute) for neural_net in neural_nets]
    labels = [neural_net.label for neural_net in neural_nets]
    
    # create bar chart
    bars = ax.bar(labels, values, color = ['C0', 'C1', 'C2'])
    
    # loop over each bar
    for bar in bars:

        # add text with height of bar above bar
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"{height:4g}",
                ha = 'center', va = 'bottom', fontsize = Inputs.fontsize)
    
    # configure plot
    ax.set_xlabel("Neural Network Type", fontsize = Inputs.fontsize)
    ax.set_ylabel(f"{ylabel}", fontsize = Inputs.fontsize)
    ax.set_title(f"{title}", fontsize = Inputs.fontsize_title)
    ax.grid(axis='y', alpha = 0.3)
    
    plt.tight_layout()

if __name__ == "__main__":
    """Runs on script execution."""
    # run main
    main()

    # show all plots
    plt.show()
