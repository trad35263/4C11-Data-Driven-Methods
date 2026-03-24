# import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import LogLocator
import torch
import torch.nn as nn
import scipy
from time import perf_counter as timer

# import modules for debugging
import psutil
import os

# import namespace of other python files
import utils

# Inputs class
class Inputs:
    """Stores input parameters for the program."""
    # file path of folder containing data
    data_folder = (
        "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/4C11 Data-Driven Methods"
        "/Coursework 2/Coursework2/Coursework2_Problem_1"
    )
    file_name = "plate_data.mat"
    exports_folder = "exports21"

    # material properties
    E = 10             # Young's modulus (N/m^2)
    poisson_ratio = 0.3             # Poisson ratio (-)
    L = 5                           # plate width (m)
    thickness = 1                   # Plate thickness (m)

    # applied stress
    sigma_1 = 0.1                   # N/m^2
    sigma_2 = 0                     # N/m^2
    sigma_1_norm = 1
    sigma_2_norm = 0

    # neural net architectural parameters
    no_of_layers = 3
    input_size = 2

    # construct stress net layer configuration
    stress_hidden_layer_size = 400
    stress_output_size = 3
    stress_net_layers = (
        [input_size] + [stress_hidden_layer_size] * (no_of_layers - 1) + [stress_output_size]
    )

    # construct displacement net layer configuration
    displacement_hidden_layer_size = 400
    displacement_output_size = 2
    displacement_net_layers = (
        [input_size] + [displacement_hidden_layer_size] * (no_of_layers - 1)
        + [displacement_output_size]
    )

    # training variables
    no_of_epochs = 1000            # recommended: 50k - 100k
    learning_rate = 1e-4
    step_size = 50
    gamma = 0.5

    # plotting parameters
    figsize = (9, 5)
    fontsize = 12
    titlesize = 14
    dpi = 300

    # terminal output parameters
    print_epoch = 10

    # flag for
    use_measurement_data = False

# Data class
class Data:
    """Stores the input data for the program and handles the pre-processing."""
    def __init__(self):
        """Creates an instance of the Data class."""
        # load data
        path = Inputs.data_folder + "/" + Inputs.file_name
        self.data = scipy.io.loadmat(path)

        # create boundary data dictionary of pyTorch tensors
        self.boundary = {
            "left": torch.tensor(self.data["L_boundary"], dtype = torch.float64),
            "right": torch.tensor(self.data["R_boundary"], dtype = torch.float64),
            "top": torch.tensor(self.data["T_boundary"], dtype = torch.float64),
            "bottom": torch.tensor(self.data["B_boundary"], dtype = torch.float64),
            "circle": torch.tensor(self.data["C_boundary"], dtype = torch.float64),
            "x": torch.tensor(self.data["Boundary"], dtype = torch.float64, requires_grad = True)
        }

        # get ground-truth displacement solution
        self.displacements = torch.tensor(self.data["disp_data"], dtype = torch.float64)

        # get collocation points - including the boundary
        self.x_full = torch.tensor(
            self.data["p_full"], dtype = torch.float64, requires_grad = True
        )

        # collocation points - excluding the boundary
        self.x = torch.tensor(self.data["p"], dtype = torch.float64, requires_grad = True)

        # choose 50 fixed points from the truth solution, used for part (e)
        self.random_index = torch.randint(0, len(self.x_full), (50,))
        self.displacements_fixed = self.displacements[self.random_index,:]
    
        # connectivity matrix - for plotting the figure but not needed for PINN
        self.connectivity_matrix = torch.tensor(
            self.data["t"].astype(float), dtype = torch.float64
        )

        # calculate stiffness matrix via Hooke's law for plane stress - normalised by E
        E = Inputs.E       # UNUSED
        nu = Inputs.poisson_ratio
        stiffness_matrix = (
            1 / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        )
        stiffness_matrix = stiffness_matrix.unsqueeze(0)
        self.stiffness_matrix = (
            torch.broadcast_to(stiffness_matrix, (len(self.x), 3, 3))
        )

        # broadcast stiffness matrix to boundaries for batch multiplication later
        stiffness_matrix_boundary = stiffness_matrix.clone()
        self.stiffness_matrix_boundary = (
            torch.broadcast_to(stiffness_matrix_boundary, (len(self.boundary["x"]), 3, 3))
        )

        # repeat for full collocation points
        stiffness_matrix_full = stiffness_matrix.clone()
        self.stiffness_matrix_full = (
            torch.broadcast_to(stiffness_matrix_full, (len(self.x_full), 3, 3))
        )

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
        title_text = f"Convergence History\n{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params}"
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

# Define Neural Network
class DenseNet(Neural_net):
    """Stores parameters relevant to the PINN model."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, nonlinearity):
        """Creates an instance of the DenseNet class."""
        # initialise Neural_net parent class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.nonlinearity = nonlinearity

        # create list of layers
        layer_sizes = (
            [self.input_size] + [self.hidden_size] * (self.no_of_layers - 1)
            + [self.output_size]
        )

        # create module list of layers
        self.layers = nn.ModuleList()

        # loop for each layer
        for index in range(self.no_of_layers):

            # append layer with appropriate input and output size
            self.layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))

            # for all but the final layer
            if index != self.no_of_layers - 1:

                # append non-linearity
                self.layers.append(nonlinearity())

        # create empty dictionary for storing losses
        self.losses = {
            "constitutive_bulk": [],
            "constitutive_boundary": [],
            "equilibrium_x": [],
            "equilibrium_y": [],
            "left_dirichlet": [],
            "left_neumann": [],
            "bottom_dirichlet": [],
            "bottom_neumann": [],
            "right_neumann_normal": [],
            "right_neumann_shear": [],
            "top_neumann_normal": [],
            "top_neumann_shear": [],
            "circle_neumann_x": [],
            "circle_neumann_y": []
        }

        # flag for using measurement data is True
        if Inputs.use_measurement_data:

            # create empty list for measurement data residuals
            self.losses["measurement_data"] = []

        # label
        self.label = "PINN"
        self.quantities = []

        if Inputs.use_measurement_data:

            self.label += " with Measurement Data"

    def forward(self, x):
        """Propagates forwards through the neural network and calculates the output."""
        # loop for each network layer
        for layer in self.layers:

            # calculate output of layer
            x = layer(x)

        # return neural network output
        return x

    #def plot_convergence(self):
        """Plots the convergence history of the neural net during training."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)

        # plot training loss against number of epochs
        ax.plot(np.arange(self.no_of_epochs), self.training_loss, label = "Training loss")

        # configure plot
        ax.grid()
        ax.legend(loc = "center left", bbox_to_anchor = (1, 0.5), fontsize = Inputs.fontsize)
        ax.set_xlabel("Number of Epochs", fontsize = Inputs.fontsize)
        ax.set_ylabel("Training Loss", fontsize = Inputs.fontsize)
        ax.set_title("Convergence History of Neural Net", fontsize = Inputs.titlesize)
        ax.set_yscale("log")

    def plot_loss_components(self):
        """Bar chart of the most recent value for each loss component."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)

        # get list of keys and values to plot from dictionary
        keys = list(self.losses.keys())
        values = np.array([self.losses[key] for key in keys])

        # get list of x-values to create bars at and specify bar_width
        xx = np.arange(len(keys))
        bar_width = 0.2

        # get list of epoch indices to plot results at
        epochs = [int(np.ceil(len(values[0]) / 10)) - 1, int(np.ceil(len(values[0]) / 4)) - 1, len(values[0]) - 1]

        # create bars corresponding to beginning, middle and final epochs
        ax.bar(xx - bar_width, values[:, epochs[0]], width = bar_width, label = f"Epoch: {epochs[0] + 1}")
        ax.bar(xx, values[:, epochs[1]], width = bar_width, label = f"Epoch: {epochs[1] + 1}")
        ax.bar(xx + bar_width, values[:, epochs[2]], width = bar_width, label = f"Epoch: {epochs[2] + 1}")

        # configure plot
        ax.set_xticks(xx)
        ax.set_xticklabels(keys, rotation = 45, ha = "right", fontsize = Inputs.fontsize)
        ax.set_ylabel("MSE Loss Magnitude", fontsize = Inputs.fontsize)
        ax.set_yscale("log")
        ax.grid(axis = "y")
        ax.legend()
        
        # set title
        fig.suptitle(
            f"Loss Component Magnitudes\n"
            f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params} | No. of Epochs: {self.no_of_epochs}",
            fontsize = Inputs.titlesize
        )

        # tight layout
        plt.tight_layout()

        # construct filename
        filename = (
            f"plot_loss_components_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, ax, filename)

    def plot_stress(self, data, norm, displacement_net):
        """Plots a grid of subplots showing the predicted vs. actual displacements data."""
        # create plot
        width, height = Inputs.figsize
        fig, axes = plt.subplots(3, 3, figsize = (width, height * 3 / 2))

        # build triangulation
        xx = data.x_full[:, 0].detach().numpy()
        yy = data.x_full[:, 1].detach().numpy()
        connect = (data.connectivity_matrix - 1).detach().numpy()
        triangles = mtri.Triangulation(xx, yy, connect)

        # get displacements and stresses from respective neural networks
        displacements_predicted = displacement_net(norm.encode(data.x_full))
        displacements_predicted = displacements_predicted * Inputs.sigma_1 * Inputs.L / Inputs.E
        stress_predicted = self(norm.encode(data.x_full)).detach().numpy()
        stress_predicted = stress_predicted * Inputs.sigma_1

        # compute strains via autograd
        u, v = displacements_predicted[:, 0], displacements_predicted[:, 1]
        du_dx = torch.autograd.grad(
            u, data.x_full, grad_outputs = torch.ones_like(u), create_graph = True
        )[0]
        dv_dx = torch.autograd.grad(
            v, data.x_full, grad_outputs = torch.ones_like(v), create_graph = True
        )[0]

        # detach displacements_predicted
        displacements_predicted = displacements_predicted.detach().numpy()

        # calculate individual strain components
        e_11 = du_dx[:, 0].unsqueeze(1)
        e_22 = dv_dx[:, 1].unsqueeze(1)
        e_12 = (0.5 * (du_dx[:, 1] + dv_dx[:, 0])).unsqueeze(1)

        # calculat strain vector
        e = torch.cat((e_11, e_22, e_12), 1).unsqueeze(2)

        # compute stress from constitutive relation
        stress_constitutive = torch.bmm(Inputs.E * data.stiffness_matrix_full, e).squeeze(2).detach().numpy()

        # calculate error
        error = np.abs(stress_constitutive - stress_predicted)

        # calculate colour bar limits
        vmin = min(stress_constitutive.min(), stress_predicted.min())
        vmax = max(stress_constitutive.max(), stress_predicted.max())
        emax = error.max()

        # create level arrays once
        levels_stress = np.linspace(vmin, vmax, 50)
        levels_err  = np.linspace(0, emax, 50)
        levels_list = [levels_stress, levels_stress, levels_err]

        # lists of plotting parameters to loop over
        colours = ["plasma", "plasma", "Reds"]
        column_titles = ["Constitutive", "Predicted", "Error"]
        row_labels = ["x-component (Pa)", "y-component (Pa)", "Shear (Pa)"]

        # loop for each row
        for row, row_label in enumerate(row_labels):
            
            # store values to plot as intermediate variables for convenience
            const = stress_constitutive[:, row]
            pred = stress_predicted[:, row]
            diff = error[:, row]

            # group as list
            fields = [const, pred, diff]

            # loop for each column
            for column, (field, title) in enumerate(zip(fields, column_titles)):

                # get relevant axis
                ax = axes[row, column]

                # create triangulated contour plot
                cf = ax.tricontourf(
                    triangles, field, levels = levels_list[column],
                    cmap = colours[column]
                )
                fig.colorbar(cf, ax = ax, shrink = 0.85)

                # top row
                if row == 0:

                    # set title
                    ax.set_title(title, fontsize = Inputs.fontsize, fontweight = 'bold')

                # first column
                if column == 0:

                    # row label on the left of each row
                    ax.text(
                        -0.3, 0.5, f"{row_label}", transform = ax.transAxes,
                        ha = 'right', va = 'center', fontsize = Inputs.fontsize, fontweight = 'bold',
                        rotation = 90
                    )

                # configure plot
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect("equal")

        # set title and tight layout
        fig.suptitle(
            f"Constitutive vs. Predicted Stress\n"
            f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params} | No. of Epochs: {self.no_of_epochs}",
            fontsize = Inputs.titlesize
        )
        fig.tight_layout()

        # construct filename
        filename = (
            f"plot_stress_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, ax, filename)

    def plot_displacements(self, data, norm):
        """Plots a grid of subplots showing the predicted vs. actual displacements data."""
        # create plot
        fig, axes = plt.subplots(2, 3, figsize = Inputs.figsize)

        # build triangulation
        xx = data.x_full[:, 0].detach().numpy()
        yy = data.x_full[:, 1].detach().numpy()
        connect = (data.connectivity_matrix - 1).detach().numpy()
        triangles = mtri.Triangulation(xx, yy, connect)

        # get (dimensional) ground thruth displacements
        displacements_truth = data.displacements.detach().numpy()

        # get displacements as predicted by neural network and convert back to dimensional
        displacements_predicted = self(norm.encode(data.x_full)).detach().numpy()
        displacements_predicted = displacements_predicted * Inputs.sigma_1 * Inputs.L / Inputs.E

        # calculate error
        error = np.abs(displacements_truth - displacements_predicted)

        # calculate colour bar limits
        vmin = min(displacements_truth.min(), displacements_predicted.min())
        vmax = max(displacements_truth.max(), displacements_predicted.max())
        emax = error.max()

        # create level arrays once
        levels_disp = np.linspace(vmin, vmax, 50)
        levels_err  = np.linspace(0, emax, 50)
        levels_list = [levels_disp, levels_disp, levels_err]

        # lists of plotting parameters to loop over
        colours = ["viridis", "viridis", "Reds"]
        column_titles = ["Ground Truth", "Predicted", "Error"]
        row_labels  = ["x-displacement (m)", "y-displacement (m)"]
        components  = [0, 1]

        # loop for each row
        for row, (component, row_label) in enumerate(zip(components, row_labels)):
            
            truth = displacements_truth[:, component]
            pred = displacements_predicted[:, component]
            diff = error[:, component]

            fields = [truth, pred, diff]

            # loop for each column
            for column, (field, title) in enumerate(zip(fields, column_titles)):

                # get relevant axis
                ax = axes[row, column]

                # create triangulated contour plot
                cf = ax.tricontourf(
                    triangles, field, levels = levels_list[column],
                    cmap = colours[column]
                )
                fig.colorbar(cf, ax = ax, shrink = 0.85)

                # top row
                if row == 0:

                    # set title
                    ax.set_title(title, fontsize = Inputs.fontsize, fontweight = 'bold')

                # first column
                if column == 0:

                    # row label on the left of each row
                    ax.text(
                        -0.3, 0.5, f"{row_label}", transform = ax.transAxes,
                        ha = 'right', va = 'center', fontsize = Inputs.fontsize, fontweight = 'bold',
                        rotation = 90
                    )

                # configure plot
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect("equal")

        # set title and tight layout
        fig.suptitle(
            f"Ground Truth vs. Predicted Displacements\n"
            f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params} | No. of Epochs: {self.no_of_epochs}",
            fontsize = Inputs.titlesize
        )
        fig.tight_layout()

        # construct filename
        filename = (
            f"plot_displacements_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, ax, filename)

# Stress_strain class
class Stress_strain():
    """Stores information related to a stress-strain state."""
    def __init__(self, x, stress, displacement):

        # store input variables
        self.x = x
        self.stress = stress
        self.displacement = displacement

        # get displacements in x- and y-directions
        u, v = displacement[:, 0], displacement[:, 1]

        # find derivatives
        du_dx = torch.autograd.grad(
            u, x, grad_outputs = torch.ones_like(u), create_graph = True, retain_graph = True
        )[0]
        dv_dx = torch.autograd.grad(
            v, x, grad_outputs = torch.ones_like(u), create_graph = True, retain_graph = True
        )[0]

        # define strains in the orthogonal and shear directions
        e_11 = du_dx[:, 0].unsqueeze(1)
        e_22 = dv_dx[:, 1].unsqueeze(1)
        e_12 = (0.5 * (du_dx[:, 1] + dv_dx[:, 0])).unsqueeze(1)

        # define strain vector by combining components and add a trailing dimension
        e = torch.cat((e_11, e_22, e_12), 1)
        self.e = e.unsqueeze(2)

# Normaliser class
class Normaliser():
    """Stores the parameters for a simple normalisation encoder and decoder."""
    def __init__(self, x, eps = 1e-6):
        """Creates an instance of the Normaliser class."""
        # store input variables
        self.eps = eps

        # calculate mean and standard deviation
        self.mean = torch.mean(x, 0).detach()
        self.std = torch.std(x, 0).detach()

    def encode(self, x):
        """Returns the normalised value(s) of x."""
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        """Inverts the normalisation process and returns the original x-value(s)."""
        x = (x * (self.std + self.eps)) + self.mean
        return x

# main function
def main():

    # create instance of Data class
    data = Data()

    # get data normaliser function
    norm = Normaliser(data.x_full)
    
    # create neural nets with tanh activation function
    stress_net = DenseNet(
        Inputs.no_of_layers, Inputs.input_size, Inputs.stress_hidden_layer_size,
        Inputs.stress_output_size, nn.Tanh
    )
    displacement_net =  DenseNet(
        Inputs.no_of_layers, Inputs.input_size, Inputs.displacement_hidden_layer_size,
        Inputs.displacement_output_size, nn.Tanh
    )
    stress_net.no_of_epochs = Inputs.no_of_epochs
    displacement_net.no_of_epochs = Inputs.no_of_epochs

    # define loss function
    loss_function = torch.nn.MSELoss()

    # print parameters
    stress_net.print_params()
    displacement_net.print_params()

    # get total list of parameters and print user feedback
    parameters = list(stress_net.parameters()) + list(displacement_net.parameters())
    no_of_parameters = sum(p.numel() for p in parameters)
    print(f"Total number of parameters: {utils.Colours.GREEN}{no_of_parameters}{utils.Colours.END}")

    # define optimiser and scheduler - try standard implementation
    optimiser = torch.optim.Adam(parameters, lr = Inputs.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size = Inputs.step_size, gamma = Inputs.gamma
    )

    # print user feedback
    t_start = timer()
    print(
        f"Training model for {utils.Colours.GREEN}{Inputs.no_of_epochs}{utils.Colours.END} epochs!"
    )

    # loop for each epoch
    for epoch in range(Inputs.no_of_epochs):

        # reset all gradient tensors of the model parameters and normalised coordinates to zero
        optimiser.zero_grad()

        # recreate normalised coordinate tensors each epoch to avoid stale computation graphs
        x_norm = norm.encode(data.x.detach()).requires_grad_(True)
        x_boundary_norm = norm.encode(data.boundary["x"].detach()).requires_grad_(True)

        # get stress and displacement from respective neural nets and normalised x-data
        stress = stress_net(x_norm)
        displacement = displacement_net(x_norm)

        # package results in object-oriented fashion
        bulk = Stress_strain(x_norm, stress, displacement)

        # calculate augmented stress via batch matrix multiplication (bmm)
        bulk.stress_augmented = torch.bmm(data.stiffness_matrix, bulk.e).squeeze(2)

        # calculate loss derived from constitutive relationship for bulk of plate
        stress_net.losses["constitutive_bulk"].append(
            loss_function(bulk.stress_augmented, stress)
        )

        # find displacement and stress at the boundaries
        displacement_boundary = displacement_net(x_boundary_norm)
        stress_boundary = stress_net(x_boundary_norm)

        # repeat for boundary nodes
        boundary = Stress_strain(x_boundary_norm, stress_boundary, displacement_boundary)

        # define augmented stress
        boundary.stress_augmented = torch.bmm(data.stiffness_matrix_boundary, boundary.e).squeeze(2)

        # calculate loss derived from consitutive relationship
        stress_net.losses["constitutive_boundary"].append(
            loss_function(boundary.stress_augmented, stress_boundary)
        )

        # extract stress components
        stress_11 = stress[:, 0]
        stress_22 = stress[:, 1]
        stress_12 = stress[:, 2]

        # calculate derivatives of each stress component
        dstress_11_dx = torch.autograd.grad(
            stress_11, x_norm, grad_outputs = torch.ones_like(stress_11), create_graph = True, retain_graph=True
        )[0]
        dstress_22_dx = torch.autograd.grad(
            stress_22, x_norm, grad_outputs = torch.ones_like(stress_11), create_graph = True, retain_graph=True
        )[0]    
        dstress_12_dx = torch.autograd.grad(
            stress_12, x_norm, grad_outputs = torch.ones_like(stress_11), create_graph = True
        )[0]

        # enforce Newton's second law for static equilibrium
        equilibrium_x1 = dstress_11_dx[:, 0] + dstress_12_dx[:, 1]
        equilibrium_x2 = dstress_22_dx[:, 0] + dstress_12_dx[:, 1]

        # get displacements on left and bottom boundaries as calculated by neural net
        displacement_L = displacement_net(norm.encode(data.boundary["left"]))
        displacement_B = displacement_net(norm.encode(data.boundary["bottom"]))

        # get stresses on boundaries as calculated by neural net
        stress_R = stress_net(norm.encode(data.boundary["right"]))
        stress_T = stress_net(norm.encode(data.boundary["top"]))
        stress_C = stress_net(norm.encode(data.boundary["circle"]))
        stress_L = stress_net(norm.encode(data.boundary["left"]))
        stress_B = stress_net(norm.encode(data.boundary["bottom"]))

        # calculate losses assocated with stress equilibrium equations
        stress_net.losses["equilibrium_x"].append(loss_function(equilibrium_x1, torch.zeros_like(equilibrium_x1)))
        stress_net.losses["equilibrium_y"].append(loss_function(equilibrium_x2, torch.zeros_like(equilibrium_x2)))

        # calculate loss corresponding to zero horizontal displacement condition on left boundary
        stress_net.losses["left_dirichlet"].append(
            loss_function(displacement_L[:, 0], torch.zeros_like(displacement_L[:, 0]))
        )

        # calculate loss corresponding to zero shear stress on left boundary
        stress_net.losses["left_neumann"].append(
            loss_function(stress_L[:, 2], torch.zeros_like(stress_L[:, 2]))
        )

        # calculate loss corresponding to zero vertical displacement condition on bottom boundary
        stress_net.losses["bottom_dirichlet"].append(
            loss_function(displacement_B[:, 1], torch.zeros_like(displacement_B[:, 1]))
        )

        # calculate loss corresponding to zero shear stress on bottom boundary
        stress_net.losses["bottom_neumann"].append(
            loss_function(stress_B[:, 2], torch.zeros_like(stress_B[:, 2]))
        )

        # calculate loss corresponding to (unit, normalised) applied stress and zero shear stress
        # on right boundary
        stress_net.losses["right_neumann_normal"].append(
            loss_function(stress_R[:, 0], Inputs.sigma_1_norm * torch.ones_like(stress_R[:, 0]))
        )
        stress_net.losses["right_neumann_shear"].append(
            loss_function(stress_R[:, 2], torch.zeros_like(stress_R[:, 2]))
        )

        # calculate loss corresponding to (zero) applied stress and zero shear stress on top
        # boundary
        stress_net.losses["top_neumann_normal"].append(
            loss_function(stress_T[:, 1], Inputs.sigma_2_norm * torch.ones_like(stress_T[:, 1]))
        )
        stress_net.losses["top_neumann_shear"].append(
            loss_function(stress_T[:, 2], torch.zeros_like(stress_T[:, 2]))
        )

        # get unit normals corresponding to circle boundary
        circle_normals = (
            data.boundary["circle"]
            / torch.sqrt(data.boundary["circle"][0, 0]**2 + data.boundary["circle"][0, 1]**2)
        )

        # calculate loss corresponding to matrix-equation (see report) circle boundary conditions
        stress_net.losses["circle_neumann_x"].append(
            loss_function(
                stress_C[:, 0] * circle_normals[:, 0] + stress_C[:, 2] * circle_normals[:, 1],
                torch.zeros_like(stress_C[:, 0])
            )
        )
        stress_net.losses["circle_neumann_y"].append(
            loss_function(
                stress_C[:, 2] * circle_normals[:, 0] + stress_C[:, 1] * circle_normals[:, 1],
                torch.zeros_like(stress_C[:, 0])
            )
        )

        # use measurement data
        if Inputs.use_measurement_data:

            # calculate loss compared to measurement data at random fixed points
            x_fix = data.x_full[data.random_index, :]
            u_fix = displacement_net(x_fix)
            loss_fix = loss_function(u_fix, Inputs.E * data.displacements_fixed / (Inputs.sigma_1 * Inputs.L))
            stress_net.losses["measurement_data"].append(100 * loss_fix)

        # calculate total training loss
        training_loss = sum(value[-1] for value in stress_net.losses.values())

        # calculate gradients of the loss with respect to model parameters
        training_loss.backward()

        # step optimiser and scheduler
        optimiser.step()
        scheduler.step()

        # convert all loss tensors to items to prevent memory leak
        for key, value in stress_net.losses.items():

            stress_net.losses[key][-1] = value[-1].item()

        # store training loss in neural network
        stress_net.training_loss.append(training_loss.item())

        # deactivate gradient tracking
        with torch.no_grad():

            # compare neural network solution at fixed points with ground truth solution
            displacements_predicted = displacement_net(norm.encode(data.x_full))
            stress_net.test_loss.append(
                loss_function(
                    displacements_predicted,
                    Inputs.E * data.displacements / (Inputs.sigma_1 * Inputs.L)
                ).item()
            )

        # end timer
        t = timer()
        stress_net.time.append(t - t_start)

        # every Nth epoch
        if (epoch + 1) % Inputs.print_epoch == 0 or epoch == 0:

            # print user feedback
            print(
                f"Epoch: {utils.Colours.GREEN}{epoch + 1}{utils.Colours.END}, "
                f"Training loss: {utils.Colours.GREEN}{stress_net.training_loss[-1]:.4g}"
                f"{utils.Colours.END}, "
                f"Test loss: {utils.Colours.GREEN}{stress_net.test_loss[-1]:.4g}"
                f"{utils.Colours.END}"
            )

            # calculate and print memory usage to debug memory leak
            process = psutil.Process(os.getpid())
            print(
                f"Memory usage: {utils.Colours.GREEN}{process.memory_info().rss / 1e6:.1f}"
                f"{utils.Colours.END} MB, "
                f"Time taken: {utils.Colours.GREEN}{t - t_start:.4g}{utils.Colours.END} s, "
                f"Time remaining: {utils.Colours.GREEN}{(t - t_start) * (Inputs.no_of_epochs / (epoch + 1) - 1):.4g}{utils.Colours.END} s"
            )

    stress_net.plot_loss_components()
    stress_net.plot_convergence()
    displacement_net.plot_displacements(data, norm)
    stress_net.plot_stress(data, norm, displacement_net)

    return

    # plot the stress

    u_full = displacement_net(data.x_full)
    stress_full = stress_net(data.x_full)

    xx = data.x_full[:, 0].detach().numpy()
    yy = data.x_full[:, 1].detach().numpy()
    sig11 = stress_full[:, 1].detach().numpy()

    connect = (data.connectivity_matrix - 1).detach().numpy()

    triang = mtri.Triangulation(xx, yy, connect)

    u_11 = u_full[:,0].detach().numpy()

    u = u_full[:, 0]
    v = u_full[:, 1]

    dudx = torch.autograd.grad(
        u, data.x_full, grad_outputs = torch.ones_like(u), create_graph = True
    )[0]
    dvdx = torch.autograd.grad(
        v, data.x_full, grad_outputs = torch.ones_like(v), create_graph = True
    )[0]

    e_11 = dudx[:, 0].unsqueeze(1)
    e_22 = dvdx[:, 1].unsqueeze(1)
    e_12 = 0.5 * (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)

    e = torch.cat((e_11, e_22, e_12), 1)
    e = e.unsqueeze(2)

    sigma = torch.bmm(data.stiffness_matrix_full, e).squeeze(2)

    # configure plot
    plt.figure(2)
    plt.clf()
    plt.tricontourf(triang, sigma[:, 0].detach().numpy())
    plt.colorbar()

# save_figure function
def save_figure(fig, ax, file_name):

    # save plot
    fig.savefig(
        f"{Inputs.exports_folder}/{file_name.replace(' ', '_')}.png",
        dpi = Inputs.dpi, bbox_inches = "tight"
    )

# upon script execution
if __name__ == "__main__":

    # set default datatype and device
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")

    # run main
    main()

    # show all plots
    plt.show()
