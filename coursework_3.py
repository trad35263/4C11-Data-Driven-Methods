# import modules
import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, LogFormatterSciNotation
from time import perf_counter as timer
import math

# import pytorch
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# import CPU monitoring modules
import psutil
import os

# import geometry modules for plotting ellipses
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# import scientific colours module
from cmcrameri import cm
from matplotlib.colors import LinearSegmentedColormap

# colours modules
import matplotlib.colors as mcolors
import colorsys

# import utils (contains Colours class)
import utils

# Inputs class
class Inputs:
    """Stores input parameters for the program."""
    # file locations
    data_folder = "../Coursework 3"
    file_name = "viscodata_3mat.mat"
    export_folder = "exports3"

    # number of training and test datapoints
    training_points = 320

    # seed for random number generation
    rng_seed = 42

    # down sampling factor to reduce temporal resolution
    sampling = 8

    # default hyperparameters
    no_of_layers = 6
    input_size = 1
    hidden_size = 8
    output_size = 1
    no_of_hidden_layers = 3
    channel_width = 16
    dropout = 0.01

    # learning parameters
    no_of_epochs = 500
    batch_size = 40
    learning_rate = 1e-3
    step_size = 50
    gamma = 0.5
    weight_decay = 1e-4

    # factor by which to multiply training dataset size via data augmentation
    augmentation_factor = 2

    # option for early stopping
    early_stopping = False
    patience = 50

    # criterion for "correct" solution (in standard deviations)
    delta = 0.5

    # plotting parameters
    figsize = (10, 6)
    fontsize = 12
    titlesize = 14
    dpi = 300

    # terminal output parameters
    print_epoch = 10

    # flag for showing plots
    show_plots_flag = False

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
            f"{utils.Colours.GREEN}{no_of_epochs}{utils.Colours.END} epochs on "
            f"{utils.Colours.GREEN}{len(training_loader) * training_loader.batch_size}"
            f"{utils.Colours.END} datapoints..."
        )
        t_start = timer()
        best_test_loss = 1e9

        # loop for each epoch
        for epoch in range(no_of_epochs):

            # set neural net to train and training loss to zero
            self.train(True)
            training_loss = 0
            training_error = 0

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
                    stress_predicted[:, index], hidden_state = self(
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

                # model errors v2
                true_points = np.stack([input, target], axis = -1)
                pred_points = np.stack([input, stress_predicted.detach().numpy()], axis = -1)

                # calculate distances from ground truth to predicted datapoints
                distances_fwd = np.array([
                    cKDTree(true_points[i]).query(pred_points[i])[0]
                    for i in range(true_points.shape[0])
                ])

                # calculate distances from predicted to ground truth datapoints
                distances_bwd = np.array([
                    cKDTree(pred_points[i]).query(true_points[i])[0]
                    for i in range(true_points.shape[0])
                ])

                # get number of passed elements and test error
                passed = (
                    np.all(distances_fwd < Inputs.delta, axis = 1)
                    & np.all(distances_bwd < Inputs.delta, axis = 1)
                )
                training_error += (~passed).sum().item()

            # step scheduler
            scheduler.step()

            # enter evaluation mode and disable gradient tracking
            self.eval()
            with torch.no_grad():

                # initialise test loss as zero
                test_loss = 0
                test_error = 0

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

                    # model errors v2
                    true_points = np.stack([input, target], axis = -1)
                    pred_points = np.stack([input, stress_predicted.detach().numpy()], axis = -1)

                    # query each sample — list comprehension avoids explicit for loop
                    distances_fwd = np.array([
                        cKDTree(true_points[i]).query(pred_points[i])[0]
                        for i in range(true_points.shape[0])
                    ])

                    distances_bwd = np.array([
                        cKDTree(pred_points[i]).query(true_points[i])[0]
                        for i in range(true_points.shape[0])
                    ])

                    # get number of passed elements and test error
                    passed = (
                        np.all(distances_fwd < Inputs.delta, axis = 1)
                        & np.all(distances_bwd < Inputs.delta, axis = 1)
                    )
                    test_error += (~passed).sum().item()
                    self.test_passes = passed

            # end timer
            t = timer()

            # append training and test loss
            self.training_loss.append(training_loss / len(training_loader))
            self.test_loss.append(test_loss / len(test_loader))
            self.time.append(t - t_start)

            # append training and test error rates
            self.training_error_rate.append(training_error / (len(training_loader) * training_loader.batch_size))
            self.test_error_rate.append(test_error / (len(test_loader) * test_loader.batch_size))

            # every Nth epoch
            if (epoch + 1) % Inputs.print_epoch == 0 or epoch == 0:

                # print user feedback
                print(
                    f"Epoch: {utils.Colours.GREEN}{epoch + 1}{utils.Colours.END}, "
                    f"Training loss: {utils.Colours.GREEN}{self.training_loss[-1]:.4g}"
                    f"{utils.Colours.END}, "
                    f"Test loss: {utils.Colours.GREEN}{self.test_loss[-1]:.4g}{utils.Colours.END}\n"
                    f"Training error rate: {utils.Colours.GREEN}"
                    f"{self.training_error_rate[-1]:.4g}"
                    f"{utils.Colours.END}, "
                    f"Test error rate: {utils.Colours.GREEN}"
                    f"{self.test_error_rate[-1]:.4g}"
                    f"{utils.Colours.END}"
                )

                # calculate and print memory usage to debug memory leak
                process = psutil.Process(os.getpid())
                print(
                    f"Memory usage: {utils.Colours.GREEN}{process.memory_info().rss / 1e6:.1f}"
                    f"{utils.Colours.END} MB, "
                    f"Time taken: {utils.Colours.GREEN}{t - t_start:.4g}{utils.Colours.END} s, "
                    f"Time remaining: {utils.Colours.GREEN}{(t - t_start) * (self.no_of_epochs / (epoch + 1) - 1):.4g}{utils.Colours.END} s, "
                )

            # early stopping option is active
            if Inputs.early_stopping:

                # test loss is the lowest yet
                if self.test_loss[-1] < best_test_loss:

                    # store best test loss and corresponding epoch
                    best_test_loss = self.test_loss[-1]
                    best_epoch = epoch + 1

                elif epoch + 1 - best_epoch >= Inputs.patience:

                    # store truncated number of epochs and end training
                    self.no_of_epochs = epoch + 1
                    print(
                        f"{utils.Colours.YELLOW}Stopping training early after {epoch + 1} epochs."
                        f"{utils.Colours.END}"
                    )
                    break

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
        ax.set_ylabel("Final MSE Loss", fontsize = Inputs.fontsize)
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

            # append quantities relevant to network to title text
            title_text += f" | {quantity[1]}: {getattr(self, quantity[0])}"
        
        # append number of training datapoints
        title_text += f"\nTraining datapoints: {Inputs.training_points * Inputs.augmentation_factor}"
        
        # set title
        ax.set_title(title_text, fontsize = Inputs.titlesize)

        # set y-axis ticks IMPORTANT
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5, subs=[1, 2, 5]))

        # tight layout
        fig.tight_layout()

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

        # add a dropout module
        self.dropout_layer = nn.Dropout(p = dropout)

        # lists for storing accuracy values
        self.training_error_rate = []
        self.test_error_rate = []

        # give label for plotting
        self.label = "RNO"
        self.quantities = [
            ["no_of_hidden_layers", "Hidden Layers"],
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

            # after a non-linear activation
            if isinstance(hidden_layer, nn.SELU):

                # apply dropout
                dh_dt = self.dropout_layer(dh_dt)

        # update hidden state via Euler integration
        hidden_state = dh_dt * self.dt + hidden_state

        # get current state of network
        x = torch.cat((previous_strain, (previous_strain - current_strain) / self.dt, hidden_state), 1)

        # loop for each layer
        for layer in self.layers:

            # pass intermediate tensor through layer
            x = layer(x)

            # after a non-linear activation
            if isinstance(layer, nn.SELU):

                # apply dropout
                x = self.dropout_layer(x)

        # return output
        output = x.squeeze(1)
        return output, hidden_state

    def plot_error_rate(self, emin = None, emax = None):
        """Plots the error rate history over the training process."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)
        
        # plot training and test loss against epoch
        ax.plot(np.arange(self.no_of_epochs), self.training_error_rate, label = "Training")
        ax.plot(np.arange(self.no_of_epochs), self.test_error_rate, label = "Test")

        # configure primary axis
        ax.grid()
        ax.set_xlabel("No. of Epochs", fontsize = Inputs.fontsize)
        ax.set_ylabel("Error Rate", fontsize = Inputs.fontsize)

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

        # axis limits are provided
        if emin is not None and emax is not None:

            # set axis limits
            ax.set_ylim(emin, emax)

        # determine title text
        title_text = f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params}"
        for quantity in self.quantities:

            # append quantities relevant to network to title text
            title_text += f" | {quantity[1]}: {getattr(self, quantity[0])}"

        # append number of training datapoints
        title_text += f"\nTraining datapoints: {Inputs.training_points * Inputs.augmentation_factor}"
        
        # set title
        ax.set_title(title_text, fontsize = Inputs.titlesize)

        # tight layout
        fig.tight_layout()

        # construct filename
        filename = (
            f"plot_error_rate_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, ax, filename)

    def plot_predictions(self, test_set, loss_function, input_normaliser, output_normaliser, plots = 6):
        """Plots the ground truth stress values against those predicted by the RNO."""
        # determine number of rows and columns
        for i in range(math.isqrt(plots), 0, -1):
            if plots % i == 0:
                rows = i
                break

        # calculate number of columns
        columns = int(plots / rows)
        
        # create plot
        width, height = Inputs.figsize
        fig, axes = plt.subplots(rows, columns, figsize = (width * columns / 3, height * rows / 2))

        # get 6 random indices of test data
        rng = np.random.default_rng(Inputs.rng_seed)
        indices = rng.choice(len(test_set), size = plots, replace = False)
        indices.sort()

        # initialise fresh hidden state
        hidden_state = torch.zeros(plots, self.hidden_size)

        # preallocate tensor of stress predictions
        stress_predicted = torch.zeros(plots, test_set[0][0].shape[0])

        # get inputs and targets from selected indices of test set
        input, target = test_set[indices][0], test_set[indices][1]

        # apply "warm start" where stress state at t = 0 is given by initial conditions
        stress_predicted[:, 0] = target[:, 0]

        # set neural net to test mode and deactivate gradient tracking
        self.eval()
        with torch.no_grad():

            # loop for each time step (starting at t = dt)
            for index in range(1, input.shape[1]):

                # propagate through RNO to determine stress prediction at next time step
                stress_predicted[:, index], self.hidden_state = self(
                    input[:, index].unsqueeze(1), input[:, index - 1].unsqueeze(1),
                    hidden_state
                )

            # calculate loss for each sample
            losses = [loss_function(x, y).item() for (x, y) in zip(stress_predicted, target)]

        # get list of bands corresponding to passing region in normalised stress-strain space
        bands = [
            unary_union([
                Point(strain, stress).buffer(Inputs.delta)
                for strain, stress in zip(input.detach().numpy()[i], target.detach().numpy()[i])
            ])
            for i in range(input.shape[0])
        ]

        # convert tensors to numpy arrays
        input = input_normaliser.decode(input)
        target = output_normaliser.decode(target)
        stress_predicted = output_normaliser.decode(stress_predicted)

        # calculate plot axis limits
        x_min, x_max = input.min(), input.max()
        y_min = min(target.min(), stress_predicted.min())
        y_max = max(target.max(), stress_predicted.max())
        margin = 0.1

        # loop for each row of subplots
        for i, row in enumerate(axes):

            # set y-axis label for LH column of plots
            row[0].set_ylabel("Stress (Pa)")

            # loop for each subplot in row
            for j, ax in enumerate(row):

                # plot ground truth and predicted solutions
                ax.plot(input[i * len(row) + j, :], target[i * len(row) + j, :], marker = '.', markersize = 2)
                ax.plot(
                    input[i * len(row) + j, :], stress_predicted[i * len(row) + j, :],
                    marker = '.', markersize = 2,
                    label = f"Loss: {losses[i * len(row) + j]:.3g}"
                )

                # get band
                band = bands[i * len(row) + j]

                # handle case where band is non-contiguous
                if band.geom_type == "Polygon":
                    geoms = [band]
                else:
                    geoms = list(band.geoms)

                # loop for each contiguous geometry
                for geom in geoms:

                    # get x- and y-coordinates
                    xs, ys = geom.exterior.xy

                    # scale geometry back to original space
                    scaled_geom = shapely.affinity.scale(
                        geom,
                        xfact=input_normaliser.std.item(),
                        yfact=output_normaliser.std.item(),
                        origin=(0, 0)
                    )

                    # add patch
                    patch = geom_to_patch(
                        scaled_geom,
                        alpha=0.2,
                        facecolor="green" if self.test_passes[indices][i * len(row) + j] else "red",
                        edgecolor="none"
                    )
                    ax.add_patch(patch)

                # configure plot
                ax.set_title(
                    f"Sample {indices[i * len(row) + j] + 1}", fontsize = Inputs.fontsize,
                    fontweight = "bold"
                )
                ax.set_xlim(x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
                ax.set_ylim(y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))
                ax.set_box_aspect(1)
                ax.grid()
                ax.legend()

                # subplot is in bottom row
                if i == len(axes) - 1:

                    # set x-axis label for bottom row of plots
                    ax.set_xlabel("Strain")

        # determine title text
        title_text = (
            f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params} | "
            f"No. of Epochs: {self.no_of_epochs}"
        )
        for quantity in self.quantities:

            title_text += f" | {quantity[1]}: {getattr(self, quantity[0])}"

        title_text += f"\nTraining datapoints: {Inputs.training_points * Inputs.augmentation_factor}"
        
        # set title
        fig.suptitle(title_text, fontsize = Inputs.titlesize)

        # tight layout
        fig.tight_layout()

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
        save_figure(fig, ax, filename)

    def plot_loss_against_time(self, test_set, ymin = None, ymax = None):
        """Plots the cumulative mean MSE loss for each test case against time."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)

        # initialise fresh hidden state
        hidden_state = torch.zeros(len(test_set), self.hidden_size)

        # preallocate tensor of stress predictions
        stress_predicted = torch.zeros(len(test_set), test_set[0][0].shape[0])

        # get inputs and targets from selected indices of test set
        input, target = test_set.tensors

        # apply "warm start" where stress state at t = 0 is given by initial conditions
        stress_predicted[:, 0] = target[:, 0]

        # set neural net to test mode and deactivate gradient tracking
        self.eval()
        with torch.no_grad():

            # loop for each time step (starting at t = dt)
            for index in range(1, input.shape[1]):

                # propagate through RNO to determine stress prediction at next time step
                stress_predicted[:, index], self.hidden_state = self(
                    input[:, index].unsqueeze(1), input[:, index - 1].unsqueeze(1),
                    hidden_state
                )

        # calculate loss as a function of time for each sample
        loss_matrix = (stress_predicted.detach().numpy() - target.detach().numpy())**2

        # get array of steps and cumulative mean for each sample
        steps = np.arange(1, loss_matrix.shape[1] + 1).reshape(1, -1)
        cumulative_mean = np.cumsum(loss_matrix, axis = 1) / steps

        # sort by final cumulative mean value, highest to lowest
        sort_indices = np.argsort(cumulative_mean[:, -1])[::-1]
        cumulative_mean = cumulative_mean[sort_indices]

        #from matplotlib import colormaps
        #print(f"list(colormaps): {list(colormaps)}")

        # get list of colours
        colours = plt.cm.viridis(np.linspace(0, 1, len(loss_matrix)))
        #colours = cm.devon(np.linspace(0, 1, len(loss_matrix)))

        # loop for each test sample
        for index, losses in enumerate(loss_matrix):

            # plot cumulative mean
            ax.plot(
                np.arange(len(losses)), cumulative_mean[index], color = colours[index], 
                linewidth = 1,
                alpha = 0.5
            )

        # configure plot
        ax.grid()
        ax.set_xlabel("Time Step", fontsize = Inputs.fontsize)
        ax.set_ylabel("Cumulative Mean MSE Test Loss", fontsize = Inputs.fontsize)
        ax.set_yscale("log")

        # axis limits are provided
        if ymin is not None and ymax is not None:

            # set axis limits
            ax.set_ylim(ymin, ymax)

        # determine title text
        title_text = (
            f"{self.label} | Layers: {self.no_of_layers} | Parameters: {self.no_of_params} | "
            f"No. of Epochs: {self.no_of_epochs}"
        )
        for quantity in self.quantities:

            title_text += f" | {quantity[1]}: {getattr(self, quantity[0])}"

        title_text += f"\nTraining datapoints: {Inputs.training_points * Inputs.augmentation_factor}"

        ax.set_title(title_text, fontsize = Inputs.titlesize)

        # construct filename
        filename = (
            f"plot_loss_against_time_{self.label}_layers_{self.no_of_layers}_parameters_"
            f"{self.no_of_params}_epochs_{self.no_of_epochs}"
        )
        for quantity in self.quantities:

            filename += f"_{quantity[0]}_{getattr(self, quantity[0])}"

        # replace all decimal points in the file name
        filename = filename.replace(".", "_")

        # save figure
        save_figure(fig, ax, filename)

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
        return x.detach().numpy()

def augment_data(data_set, alpha = 0.2):
    """Applies a principled global mixup using a Beta distribution for interpolation weights."""
    # no data augmentation is required
    if Inputs.augmentation_factor <= 1:

        # return dataset as is
        return data_set
        
    # setup local Generators for reproducibility
    rng_np = np.random.default_rng(seed = Inputs.rng_seed)
    rng_torch = torch.Generator()
    rng_torch.manual_seed(Inputs.rng_seed)

    # extract raw tensors
    inputs_all, targets_all = data_set.tensors
    N, _ = inputs_all.shape
    
    # store as lists
    all_inputs = [inputs_all]
    all_targets = [targets_all]

    # loop for each new version of the dataset required
    for _ in range(Inputs.augmentation_factor - 1):

        # create a random permutation to find partners for every sample
        perm = torch.randperm(N, generator = rng_torch)
        
        # draw weights from a Beta distribution for each pair of samples
        r = rng_np.beta(alpha, alpha, size = (N, 1))
        r = torch.from_numpy(r).float()
        
        # apply linear combination
        mixed_inputs = r * inputs_all + (1 - r) * inputs_all[perm]
        mixed_targets = r * targets_all + (1 - r) * targets_all[perm]
        
        # append to dataset
        all_inputs.append(mixed_inputs)
        all_targets.append(mixed_targets)

    # concatenate and return
    final_input = torch.cat(all_inputs, dim=0)
    final_target = torch.cat(all_targets, dim=0)
    return Data.TensorDataset(final_input, final_target)

def geom_to_patch(geom, **kwargs):
    """Converts a Shapely Polygon (with holes) to a matplotlib PathPatch."""
    
    def ring_to_path_coords(ring):
        coords = np.array(ring.coords)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
        return coords, codes

    all_coords = []
    all_codes = []

    # exterior ring
    coords, codes = ring_to_path_coords(geom.exterior)
    all_coords.append(coords)
    all_codes.extend(codes)

    # interior rings (holes)
    for interior in geom.interiors:
        coords, codes = ring_to_path_coords(interior)
        all_coords.append(coords)
        all_codes.extend(codes)

    path = Path(np.concatenate(all_coords), all_codes)
    return PathPatch(path, **kwargs)

# plot_nets function
def plot_nets(nets):
    """Creates a scatter plot of the different neural network performances."""
    # helper function
    def adjust_lightness(color, amount):
        """amount < 1 darkens, amount > 1 lightens. 1.0 = unchanged."""
        c = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    # create plots
    fig, axes = plt.subplots(2, 1, figsize = Inputs.figsize)

    # Build mappings
    unique_layers = sorted(set(net.no_of_layers for net in nets))
    unique_dropouts = sorted(set(net.dropout for net in nets))

    # set base colours
    base_colours = {layers: f"C{i}" for i, layers in enumerate(unique_layers)}
    swap = {"C1": "C3", "C3": "C1"}
    base_colours = {layers: swap.get(colour, colour) for layers, colour in base_colours.items()}

    # set "lightness" values
    lightness_values = {d: 1 + (i / max(len(unique_dropouts) - 1, 1)) * 0.7
                    for i, d in enumerate(unique_dropouts)}

    # group nets by label
    layers_dropouts = []
    labels = []
    colours = []

    # loop for each neural network
    for net in nets:

        # dropout is not yet stored list
        if not (net.no_of_layers, net.dropout) in layers_dropouts:

            # append dropout, label and a new colour
            layers_dropouts.append((net.no_of_layers, net.dropout))
            labels.append(f"Layers = {net.no_of_layers}, p = {net.dropout}")
            colours.append(adjust_lightness(base_colours[net.no_of_layers], lightness_values[net.dropout]))

    # for each dropout value found
    for (no_of_layers, dropout), label, colour in zip(layers_dropouts, labels, colours):

        # form group of neural networks and plot test loss and test error rate
        group = [net for net in nets if net.no_of_layers == no_of_layers and net.dropout == dropout]
        axes[0].scatter(
            [net.no_of_params for net in group],
            [net.test_loss[-1] for net in group],
            color = colour,
            label = label,
            s = 60 if dropout == 0 else np.pi * 15,
            marker = 'o' if dropout == 0 else 's'
        )
        axes[1].scatter(
            [net.no_of_params for net in group],
            [net.test_error_rate[-1] for net in group],
            color = colour,
            label = label,
            s = 60 if dropout == 0 else np.pi * 15,
            marker = 'o' if dropout == 0 else 's'
        )

    # configure plot
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel('Number of Parameters', fontsize = Inputs.fontsize)
    axes[0].set_ylabel('Final MSE Test Loss', fontsize = Inputs.fontsize)
    axes[1].set_ylabel('Final Test Error Rate', fontsize = Inputs.fontsize)
    fig.suptitle(
        "Neural Networks Comparison"
        f"\nTraining datapoints: {Inputs.training_points * Inputs.augmentation_factor}",
        fontsize = Inputs.titlesize)
    axes[0].grid()
    axes[1].grid()
    axes[1].legend(fontsize = Inputs.fontsize)

    # set y-axis ticks IMPORTANT
    axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=5, subs=[1, 2, 5]))

    # tight layout
    fig.tight_layout()

    # save figure
    save_figure(fig, axes, "plot_nets")

# save_figure function
def save_figure(fig, ax, filename):

    # save plot
    fig.savefig(f"{Inputs.export_folder}/{filename}", dpi = Inputs.dpi)

    if Inputs.show_plots_flag == False:

        plt.close()

# main function
def main():

    print(torch.get_num_threads())         # Should print 4
    print(torch.get_num_interop_threads()) # Should print 4

    # set pyTorch and numpy seeds
    torch.manual_seed(Inputs.rng_seed)

    # load data
    data_loader = MatReader(Inputs.data_folder + "/" + Inputs.file_name)

    # get input and output data
    n_total = data_loader.data["epsi_tol"].shape[0]
    data_input = data_loader.read_field("epsi_tol").contiguous().view(n_total, -1)
    data_output = data_loader.read_field("sigma_tol").contiguous().view(n_total, -1)

    # down sample the data to reduce temporal resolution
    data_input = data_input[:, 0::Inputs.sampling]
    data_output = data_output[:, 0::Inputs.sampling]

    # shuffle input and output data randomly
    rng = np.random.default_rng(Inputs.rng_seed)
    indices = rng.choice(len(data_input), size = len(data_input), replace = False)
    data_input = data_input[indices]
    data_output = data_output[indices]

    # normalise data (based on training data only)
    input_normaliser = Normaliser(data_input[:Inputs.training_points, :])
    output_normaliser = Normaliser(data_output[:Inputs.training_points, :])
    data_input = input_normaliser.encode(data_input)
    data_output = output_normaliser.encode(data_output)

    # set RNG seed for DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(Inputs.rng_seed)

    # create training data loader
    training_set = Data.TensorDataset(
        data_input[:Inputs.training_points, :], data_output[:Inputs.training_points, :]
    )
    training_set = augment_data(training_set)
    training_loader = Data.DataLoader(
        training_set, Inputs.batch_size, shuffle = True, generator = g,
        num_workers = 0, persistent_workers = False
    )

    # create test data loader with batch size equal to entire test dataset
    test_set = Data.TensorDataset(
        data_input[Inputs.training_points:, :], data_output[Inputs.training_points:, :]
    )
    test_loader = Data.DataLoader(test_set, len(test_set), shuffle = False)

    # define time increment to use
    dt = 1 / (data_output.shape[1] - 1)

    # define loss function
    loss_function = nn.MSELoss()

    # get list of networks to train
    nets = []
    dropouts = [0]
    channel_widths = [2, 4, 8, 16, 32]
    layers = [4, 6, 8]
    
    # loop for each number of layers
    for layer in layers:

        # loop for each channel width
        for channel_width in channel_widths:
                        
            # loop for each dropout
            for p in dropouts:

                # create RNO and store in list
                nets.append(
                    RNO(
                        layer, Inputs.input_size, Inputs.hidden_size, Inputs.output_size,
                        Inputs.no_of_hidden_layers, channel_width, p, dt
                    )
                )

    # create neural network instance
    """nets = [RNO(
        Inputs.no_of_layers, Inputs.input_size, Inputs.hidden_size, Inputs.output_size,
        Inputs.no_of_hidden_layers, Inputs.channel_width, Inputs.dropout, dt
    )]"""

    # loop for each network in list
    for net in nets:

        # print number of parameters
        net.print_params()

        # define optimiser and scheduler
        optimiser = torch.optim.AdamW(
            net.parameters(), lr = Inputs.learning_rate, weight_decay = Inputs.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size = Inputs.step_size, gamma = Inputs.gamma
        )

        # train neural network
        net.training_loop(
            Inputs.no_of_epochs, training_loader, test_loader, loss_function, optimiser, scheduler
        )

    # determine axis limits for convergence plot
    ymin = np.min([[net.test_loss, net.training_loss] for net in nets])
    ymax = np.max([[net.test_loss, net.training_loss] for net in nets])

    # determine axis limits for error rate plot
    emin = np.min([[net.test_error_rate, net.training_error_rate] for net in nets])
    emax = np.max([[net.test_error_rate, net.training_error_rate] for net in nets])
    margin = 0.1

    # loop for each network in list
    for net in nets:

        # plot convergence history
        net.plot_convergence(0.5 * ymin, 1.2 * ymax)
        net.plot_error_rate(emin - margin * (emax - emin), emax + margin * (emax - emin))

        # plot predictions
        net.plot_predictions(test_set, loss_function, input_normaliser, output_normaliser)
        net.plot_loss_against_time(test_set)

    plot_nets(nets)

# upon script execution
if __name__ == "__main__":

    #torch.set_num_threads(2)          # Intra-op parallelism (within a single op, e.g. matmul)
    #torch.set_num_interop_threads(2)  # Inter-op parallelism (parallel independent ops)

    # run main()
    main()

    # show all plots
    if Inputs.show_plots_flag:
        plt.show()
