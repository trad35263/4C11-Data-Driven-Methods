# import modules
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random

# import pytorch
import torch
import torch.nn as nn
import torch.utils.data as Data

# import neural network classes
from neural_nets import Const_Net, Dataset

# import colours
from utils import Colours as colours

# create Inputs class
class Inputs:
    """Stores high-level inputs for the program and resultant neural net."""
    # file path of folder containing material data
    data_folder = "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/4C11 Data-Driven Methods/Project_1/Problem_1_student/Data"
    file_name = "Material_C.mat"

    # batch size
    batch_size = 32

    # number of training datapoints
    ntrain = 880

    # number of training epochs
    no_of_epochs = 200

    # global figure size and fontsize
    figsize = (7, 5)
    fontsize = 12
    fontsize_title = 14

    # global neural network parameters
    input_size = 6      # overwritten based on input data
    hidden_size = 64
    output_size = 6     # overwritten based on input data
    no_of_layers = 3

    # neural network parameters (specific to certain architectures)
    dropout = 0.25
    nonlinearity = "relu"

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

# main function
def main():
    """Runs on script execution."""
    # set up data reader
    path = Inputs.data_folder + "/" + Inputs.file_name
    data_reader = MatRead(path)
    print(f"Reading file {colours.GREEN}{path}{colours.END}...")

    # read data
    strain_data = data_reader.get_strain()
    stress_data = data_reader.get_stress()

    # update input and output size dynamically (for material C)
    Inputs.input_size = strain_data.shape[2]
    Inputs.output_size = strain_data.shape[2]

    # shuffle data pseudo-randomly
    rng = np.random.default_rng(seed = 42)
    shuffled_indices = rng.permutation(len(strain_data))
    strain_data = strain_data[shuffled_indices]
    stress_data = stress_data[shuffled_indices]

    # store data in Dataset classes
    strain = Dataset(strain_data, Inputs.ntrain)
    stress = Dataset(stress_data, Inputs.ntrain)

    # print user feedback
    print(f"Number of training samples: {colours.GREEN}{Inputs.ntrain}{colours.END}")
    print(
        f"Number of test samples: {colours.GREEN}{len(strain.data) - Inputs.ntrain}{colours.END}"
    )

    # create data loader for training data
    train_set = Data.TensorDataset(
        strain.training_data_normalised, stress.training_data_normalised
    )
    train_loader = Data.DataLoader(train_set, Inputs.batch_size, shuffle = True)

    # create data loader for test data
    test_set  = Data.TensorDataset(strain.test_data_normalised, stress.test_data_normalised)
    test_loader = Data.DataLoader(test_set, Inputs.batch_size, shuffle = False)

    # define loss function
    loss_function = Lossfunc()

    # create neural network
    neural_net = Const_Net(
        Inputs.no_of_layers, Inputs.input_size, Inputs.hidden_size, Inputs.output_size,
        Inputs.dropout, Inputs.nonlinearity
    )

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
    neural_net.plot_convergence(Inputs.figsize, Inputs.fontsize, Inputs.fontsize_title, Inputs.file_name)

    # plot random stress-strain curve
    plot_stress_strain(strain, stress, neural_net)

    # apply ramp strain inputs components
    plot_stress_strain(strain, stress, neural_net, 1)

    if Inputs.input_size > 1:
        plot_stress_strain(strain, stress, neural_net, 2)
        plot_stress_strain(strain, stress, neural_net, 4)

# plot_stress_strain function
def plot_stress_strain(strain, stress, neural_net, ramp_component = None):
    
    if ramp_component == None:

        # choose a random test sample index
        sample_index = random.randint(0, len(strain.test_data))

        # get encoded strain (network input)
        strain_input = strain.test_data_normalised[sample_index:sample_index + 1]

    else:

        # get maximum strain value in strain.test_data
        max_strain = strain.test_data.max().item()

        # create tensor with np.linspace(0, max, strain.test_data.shape[1]) in [: : ramp_component - 1]
        strain_input = torch.zeros(1, strain.test_data.shape[1], strain.test_data.shape[2])
        ramp_up = np.linspace(0, max_strain, strain.test_data.shape[1] // 2)
        ramp_down = np.linspace(max_strain, 0, strain.test_data.shape[1] - strain.test_data.shape[1] // 2)
        strain_input[0, :, ramp_component - 1] = torch.tensor(np.concatenate([ramp_up, ramp_down]))
        
        # normalise the ramp input
        strain_input = strain.normalise(strain_input)

        sample_index = None

    # set gradient tracking to zero and get predicted stress
    neural_net.eval()
    with torch.no_grad():
        stress_pred_encode = neural_net(strain_input)

    # decode prediction back to physical scale
    stress_prediction = stress.inverse_normalise(stress_pred_encode)
    stress_prediction = stress_prediction.squeeze(0)

    # get ground truth stress (unnormalised)
    if ramp_component == None:
        stress_true = stress.test_data[sample_index]
        stress_true_np = stress_true.numpy()

    # store stress and strains as numpy arrays for plotting
    strain_np = strain.inverse_normalise(strain_input).squeeze(0).numpy()
    stress_prediction_np = stress_prediction.numpy()

    # if input size is 1
    if Inputs.input_size == 1:

        # create plot of only one component
        fig, ax = plt.subplots(figsize = Inputs.figsize)
        axes = [ax]

    # if input size is 6
    else:

        # create 2 x 3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize = Inputs.figsize, sharex = True, sharey = True)
        axes = axes.flatten()

    # loop over all axes
    for index, ax in enumerate(axes):

        # configure plot
        ax.plot(strain_np[:, index], stress_prediction_np[:, index], label = "Predicted")
        if ramp_component == None:
            ax.plot(strain_np[:, index], stress_true_np[:, index], label = "True")
        ax.grid()

        # set subplot title
        ax.text(
            0.5, 1.02, f"Component {index + 1}", transform = ax.transAxes,
            ha = 'center', va = 'bottom', fontsize = Inputs.fontsize
        )

    # configure plot
    axes[-1].legend(fontsize = Inputs.fontsize)

    if ramp_component == None:

        # set plot title
        fig.text(
            0.5, 0.95,
            f"{Inputs.file_name} | Sample index: {sample_index} | No. of epochs: {Inputs.no_of_epochs} | "
            f"Layers: {Inputs.no_of_layers}",
            ha = 'center', va = 'bottom', fontsize = Inputs.fontsize
        )

    else:

        # set plot title
        fig.text(
            0.5, 0.95,
            f"{Inputs.file_name} | Strain ramp component: {ramp_component} | No. of epochs: {Inputs.no_of_epochs} | "
            f"Layers: {Inputs.no_of_layers}",
            ha = 'center', va = 'bottom', fontsize = Inputs.fontsize
        )

    fig.text(0.5, 0.05, "Strain", ha = 'center', va = 'center', fontsize = Inputs.fontsize)
    fig.text(
        0.05, 0.5, "Stress", ha = 'center', va = 'center', rotation = 'vertical',
        fontsize = Inputs.fontsize
    )
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# runs on script execution
if __name__ == "__main__":
    """"""
    # run main
    main()

    # show all plots
    plt.show()
