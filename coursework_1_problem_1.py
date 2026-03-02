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

# import colours
from utils import Colours as colours

# create Inputs class
class Inputs:
    """Stores high-level inputs for the program and resultant neural net."""
    # file path of folder contain material data
    data_folder = "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/4C11 Data-Driven Methods/Project_1/Problem_1_student/Data"

    # batch size
    batch_size = 10

    # number of training datapoints
    ntrain = 600

    # number of training epochs
    no_of_epochs = 100

    # global figure size and fontsize
    figsize = (7, 5)
    fontsize = 12

# create Outputs class
class Outputs(Inputs):
    """Stores the most important outputs of the program."""
    def __init__(
            self,
            net, test_strain, test_stress, test_strain_encode, test_stress_encode,
            strain_normaliser, stress_normaliser,
            training_loss_list, test_loss_list, t_start
        ):
        """Creates an instance of the Outputs class."""
        # store input variables
        self.net = net
        self.test_strain = test_strain
        self.test_stress = test_stress
        self.test_strain_encode = test_strain_encode
        self.test_stress_encode = test_stress_encode
        self.strain_normaliser = strain_normaliser
        self.stress_normaliser = stress_normaliser
        self.training_loss_list = training_loss_list
        self.test_loss_list = test_loss_list
        self.t_start = t_start

        # print time taken
        self.t_end = timer()
        print(f"Completed after {colours.GREEN}{self.t_end - self.t_start:.4g}{colours.END} s!")

        # print final losses
        print(f"Final training loss {colours.GREEN}{self.training_loss_list[-1]}{colours.END}")
        print(f"Final test loss {colours.GREEN}{self.test_loss_list[-1]}{colours.END}")

    def plot_convergence(self):
        """Plots the convergence history of the training."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)
        
        # plot training and test loss against epoch
        ax.plot(np.arange(self.no_of_epochs), self.training_loss_list, label = "Training")
        ax.plot(np.arange(self.no_of_epochs), self.test_loss_list, label = "Test")

        # configure plot
        ax.grid()
        ax.legend(fontsize = self.fontsize)
        ax.set_xlabel("No. of Epochs", fontsize = self.fontsize)
        ax.set_ylabel("Loss", fontsize = self.fontsize)
        ax.set_yscale("log")

    def plot_stress_strain(self):
        # choose a random test sample index
        sample_index = random.randint(0, len(self.test_strain))

        # get encoded strain (network input)
        strain_input = self.test_strain_encode[sample_index:sample_index + 1]

        # set gradient tracking to zero and get predicted stress
        self.net.eval()
        with torch.no_grad():
            stress_pred_encode = self.net(strain_input)

        # decode prediction back to physical scale
        stress_prediction = self.stress_normaliser.inverse_transform(stress_pred_encode)
        stress_prediction = stress_prediction.squeeze(0)

        # get ground truth stress (unnormalised)
        stress_true = self.test_stress[sample_index]

        # store stress and strains as numpy arrays for plotting
        strain_np = self.test_strain[sample_index].numpy()
        stress_true_np = stress_true.numpy()
        stress_prediction_np = stress_prediction.numpy()

        # create 2 x 3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize = Inputs.figsize, sharex = True, sharey = True)
        axes = axes.flatten()

        # loop over all axes
        for index, ax in enumerate(axes):

            # configure plot
            ax.plot(strain_np[:, index], stress_true_np[:, index], label = "True")
            ax.plot(strain_np[:, index], stress_prediction_np[:, index], label = "Predicted")
            ax.grid()

            # set subplot title
            ax.text(
                0.5, 1.02, f"Component {index + 1}", transform = ax.transAxes,
                ha = 'center', va = 'bottom', fontsize = self.fontsize
            )

        # configure plot
        axes[-1].legend(fontsize = self.fontsize)

        # set plot title
        fig.text(
            0.5, 0.95, f"Sample index: {sample_index} | No. of epochs: {self.no_of_epochs}",
            ha = 'center', va = 'bottom', fontsize = self.fontsize
        )
        fig.text(0.5, 0.05, "Strain", ha = 'center', va = 'center', fontsize = Inputs.fontsize)
        fig.text(
            0.05, 0.5, "Stress", ha = 'center', va = 'center', rotation = 'vertical',
            fontsize = Inputs.fontsize
        )
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# create Lossfunc class
class Lossfunc(nn.Module):
    """Mean squared error loss."""
    def __init__(self):
        """Creates an instance of the Lossfunc class."""
        # calculate mean-squared error
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        """Performs forward propagation."""
        return self.mse(prediction, target)

# create MatRead class
class MatRead(object):
    """Reads the matlab data from the .mat file provided."""
    def __init__(self, file_path):
        """Creates an instance of the MatRead class."""
        # initialise input object if necessary
        super(MatRead).__init__()

        # store input file path
        self.file_path = file_path

        # read data
        self.data = h5py.File(self.file_path)

    def get_strain(self):
        """Reads the strain-data from the data object."""
        # read strain data and return as pytorch tensor
        strain = np.array(self.data['strain']).transpose(2, 0, 1)
        return torch.tensor(strain, dtype = torch.float32)

    def get_stress(self):
        """Reads the stress-data from the data object."""
        # read stress data and return as pytorch tensor
        stress = np.array(self.data['stress']).transpose(2, 0, 1)
        return torch.tensor(stress, dtype = torch.float32)

# DataNormaliser class
class DataNormaliser(object):
    """Normalises tensors with shape (N, T, C).

    Accepts either a torch.Tensor or numpy array.
    """
    def __init__(self, data, eps = 1e-8):
        """"""
        self.eps = eps
        self.fit(data)

    def fit(self, data):
        """"""
        # compute statistics across samples and time dims -> shape (1,1,C)
        self.mean = data.mean(dim = (0, 1), keepdim = True)
        
        self.std = data.std(
            dim = (0, 1), unbiased = False, keepdim = True
        ).clamp_min(self.eps)

    def transform(self, x):
        """if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)"""
        
        #if self.method == 'standard':

        return (x - self.mean) / self.std
    
        """else:
            return (x - self.min) / (self.max - self.min + self.eps)"""

    def inverse_transform(self, x):
        #if self.method == 'standard':

        return x * self.std + self.mean
    
        """else:
            return x * (self.max - self.min + self.eps) + self.min"""

# Define network your neural network for the constitutive model below
class Const_Net(nn.Module):
    """Specifies the neural network archicture.

    Input: x of shape (N, T, C) where C is number of strain components.
    Output: stress prediction of shape (N, T, C).
    """
    def __init__(
        self,
        input_size = 6,
        hidden_size = 128,
        num_layers = 2,
        dropout = 0.0,
        bidirectional = False,
        nonlinearity = "relu",
    ):
        # initialise instance of nn.Module class
        super().__init__()

        # store input variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

        # create recurrent neural network
        self.rnn = nn.RNN(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            nonlinearity = self.nonlinearity,
        )

        # create fully connected neural network
        self.fully_connected = nn.Linear(hidden_size * (2 if bidirectional else 1), input_size)

    def forward(self, x):
        """Propagates forwards through the neural network architecture."""
        # x: (N, T, C)
        # get recurrent neural network output (N, T, hidden_size)
        out, _ = self.rnn(x)

        # get fully connected neural network output (N, T, C) and return
        out = self.fully_connected(out)
        return out
    
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module

# main function
def main():
    """Runs on script execution."""
    # set data path and read stress and strain data
    path = Inputs.data_folder + '/Material_A.mat'
    data_reader = MatRead(path)
    strain = data_reader.get_strain()
    stress = data_reader.get_stress()

    # specify the training data
    train_strain = strain[:Inputs.ntrain]
    train_stress = stress[:Inputs.ntrain]

    # specify the test data
    test_strain = strain[len(strain) - Inputs.ntrain:]
    test_stress = stress[len(strain) - Inputs.ntrain:]

    print(f"Number of training samples: {colours.GREEN}{Inputs.ntrain}{colours.END}")
    print(f"Number of test samples: {colours.GREEN}{len(strain) - Inputs.ntrain}{colours.END}")

    # normalise strain data
    strain_normaliser = DataNormaliser(train_strain)
    train_strain_encode = strain_normaliser.transform(train_strain)
    test_strain_encode = strain_normaliser.transform(test_strain)

    # normalise stress data
    stress_normaliser = DataNormaliser(train_stress)
    train_stress_encode = stress_normaliser.transform(train_stress)
    test_stress_encode = stress_normaliser.transform(test_stress)

    # store number of components and timestep information separately for convenience
    ndim = strain.shape[2]          # not used
    nstep = strain.shape[1]
    dt = 1 / (nstep - 1)            # not used

    # create data loader for training data
    batch_size = Inputs.batch_size
    train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle = True)

    # create data loader for test data
    test_set  = Data.TensorDataset(test_strain_encode, test_stress_encode)
    test_loader = Data.DataLoader(test_set, batch_size, shuffle = True)

    # create neural network
    net = Const_Net()

    # calculate the number of training parameters
    no_of_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {colours.GREEN}{no_of_params}{colours.END}")

    # define loss function
    loss_function = Lossfunc()

    # define optimiser
    optimiser = torch.optim.Adam(
        net.parameters(),
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

    # print number of epochs
    print(f"Start training for {colours.GREEN}{Inputs.no_of_epochs} epochs{colours.END}...")
    t_start = timer()

    # create empty lists to store loss convergence data
    training_loss_list = []
    test_loss_list = []

    # loop for each epoch
    for epoch in range(Inputs.no_of_epochs):

        # train neural network and initialise training loss as zero
        net.train()
        training_loss = 0

        # loop through training data
        for index, (input, target) in enumerate(train_loader):

            # define forward neural network
            output = net(input)

            # calculate loss
            loss = loss_function.forward(output, target)

            # clear gradients
            optimiser.zero_grad()
            
            # compute gradients
            loss.backward()

            # update weights
            optimiser.step()

            # update training loss
            training_loss += loss.item()

        # divide through by number of samples
        training_loss /= len(test_loader)

        # enter evaluation mode and disable gradient tracking
        net.eval()
        with torch.no_grad():

            # initialise test loss as zero
            test_loss = 0

            # loop through training data
            for index, (input, target) in enumerate(test_loader):

                # define forward neural network
                output = net(input)

                # calculate loss
                loss = loss_function.forward(output, target)

                # update training loss
                test_loss += loss.item()

        # use scheduler
        scheduler.step(test_loss)

        # divide through by number of samples
        test_loss /= len(test_loader)

        # print train loss every 10 epochs
        if epoch % 10 == 0:

            print(
                f"Epoch: {epoch}, training loss: {training_loss:.4g}, "
                f"test loss: {test_loss:.4g}, time taken: {timer() - t_start:.4g} s"
            )

        # save loss
        training_loss_list.append(training_loss)
        test_loss_list.append(test_loss)

    # package results as Outputs class
    outputs = Outputs(
        net, test_strain, test_stress, test_strain_encode, test_stress_encode,
        strain_normaliser, stress_normaliser,
        training_loss_list, test_loss_list, t_start
    )
    outputs.plot_convergence()
    outputs.plot_stress_strain()

# runs on script execution
if __name__ == "__main__":
    """"""
    # run main
    main()

    # show all plots
    plt.show()
