# import pytorch
import torch
import torch.nn as nn

# import modules
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as timer

# import colours
from utils import Colours as colours

# Dataset class
class Dataset():
    """Stores a dataset with options to normalise the data."""
    def __init__(self, data, ntrain, epsilon = 1e-9):
        """Creates an instance of the Dataset class."""
        # store input variables
        self.data = data
        self.ntrain = ntrain
        self.epsilon = epsilon

        # specify the training and test data
        self.training_data = self.data[:self.ntrain]
        self.test_data = self.data[self.ntrain:]

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
        return (x - self.mean) / (self.std + self.epsilon)

    def inverse_normalise(self, x):
        """Reverses the normalisation process for a given data."""
        # calculate unnormalised value(s)
        return x * (self.std + self.epsilon) + self.mean

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
    
    def training_loop(
            self, no_of_epochs, train_loader, test_loader, loss_function, optimiser, scheduler
        ):
        """Trains the neural network."""
        # store input variables
        self.no_of_epochs = no_of_epochs

        # print number of epochs
        print(f"Start training {colours.GREEN}{self.label}{colours.END} for {colours.GREEN}{no_of_epochs} epochs{colours.END}...")
        t_start = timer()

        # loop for each epoch
        for epoch in range(no_of_epochs):

            # train neural network and initialise training loss as zero
            self.train()
            training_loss = 0

            # loop through training data
            for index, (input, target) in enumerate(train_loader):

                # define forward neural network
                output = self(input)

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
            training_loss /= len(train_loader)

            # enter evaluation mode and disable gradient tracking
            self.eval()
            with torch.no_grad():

                # initialise test loss as zero
                test_loss = 0

                # loop through test data
                for index, (input, target) in enumerate(test_loader):

                    # define forward neural network
                    output = self(input)

                    # calculate loss
                    loss = loss_function.forward(output, target)

                    # update training loss
                    test_loss += loss.item()

            # use scheduler
            scheduler.step(test_loss)

            # divide through by number of samples
            test_loss /= len(test_loader)

            print(f"Train loss: {training_loss:.6f} (computed in train mode)")
            print(f"Test loss: {test_loss:.6f} (computed in eval mode)")

            # Also check if normalizations match
            train_data_all = torch.cat([el[1] for el in train_loader])
            test_data_all = torch.cat([el[1] for el in test_loader])

            print(f"Train data range: {train_data_all.min():.4f} to {train_data_all.max():.4f}")
            print(f"Test data range: {test_data_all.min():.4f} to {test_data_all.max():.4f}")

            # print train loss every 10 epochs
            if epoch % 10 == 0:

                print(
                    f"Epoch: {epoch}, training loss: {training_loss:.4g}, "
                    f"test loss: {test_loss:.4g}, time taken: {timer() - t_start:.4g} s"
                )

            # save loss
            t = timer()
            self.training_loss.append(training_loss)
            self.test_loss.append(test_loss)
            self.time.append(t - t_start)

    def plot_convergence(self, figsize, fontsize, fontsize_title, file_name):
        """Plots the convergence history of the training and testing."""
        # create plot
        fig, ax = plt.subplots(figsize = figsize)
        
        # plot training and test loss against epoch
        ax.plot(np.arange(self.no_of_epochs), self.training_loss, label = "Training")
        ax.plot(np.arange(self.no_of_epochs), self.test_loss, label = "Test")

        # configure plot
        ax.grid()
        ax.legend(fontsize = fontsize)
        ax.set_xlabel("No. of Epochs", fontsize = fontsize)
        ax.set_ylabel("Loss", fontsize = fontsize)
        ax.set_yscale("log")
        #ax.set_ylim(5e-3, 1)

        # set title
        ax.text(
            0.5, 1.03,
            f"{file_name} | {self.label} | "
            f"Layers: {self.no_of_layers} | Parameters: {self.no_of_params}",
            transform = ax.transAxes,
            ha = 'center',
            va = 'bottom',
            fontsize = fontsize_title
        )

    def calculate_accuracy(self, data_loader):
        """Calculates accuracy on a given dataset."""
        # Enter evaluation mode
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            for input, target in data_loader:
                # Forward pass
                output = self(input)
                
                # Convert to binary predictions (threshold at 0.5)
                predictions = (output > 0.5).float()
                
                # Count correct predictions
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        # Calculate accuracy
        self.accuracy = correct / total

    def print_params(self):
        """Prints the number of parameters stored in the neural net."""
        # calculate number of parameters and print
        self.no_of_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {colours.GREEN}{self.no_of_params}{colours.END}")

# Const_Net class
class Const_Net(Neural_net):
    """Specifies the architecture for a recurrent neural network with a linear layer."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, dropout, nonlinearity):
        # initialise instance of Neural_net class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        # store label
        self.label = "Const_Net"

        # create recurrent neural network
        self.rnn = nn.RNN(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.no_of_layers - 1,
            batch_first = True,
            dropout = self.dropout,
            nonlinearity = self.nonlinearity,
        )

        # create linear layer
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # print the number of parameters
        self.print_params()

    def forward(self, x):
        """Propagates forwards through the neural network architecture."""
        # get recurrent neural network output
        out, _ = self.rnn(x)

        # get linear layer output and return
        out = self.linear(out)
        return out

# FCNN class
class FCNN(Neural_net):
    """Specifies the architecture for a fully connected neural network."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size):
        """"""
        # initialise instance of Neural_net class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store label
        self.label = "FCNN"

        # determine array of layers sizes needed
        hidden_sizes = self.hidden_size * np.ones(2 * (self.no_of_layers - 1))
        layer_sizes = np.array(
            [self.input_size] + [int(size) for size in hidden_sizes] + [self.output_size],
            dtype = int
        )
        layer_sizes = layer_sizes.reshape(-1, 2)

        # create empty list of layers objects
        layers = []

        # loop for each layer
        for layer_size in layer_sizes:

            # create linear layer and store it in the list
            layers.append(nn.Linear(*layer_size))
            layers.append(nn.ReLU())
        
        # remove unwanted ReLU after output layer
        layers = layers[:-1]

        # combine into Sequential
        self.net = nn.Sequential(*layers)

        # print the number of parameters
        self.print_params()

# Residual_block class
class Residual_block(nn.Module):
    """Component of a ResNet used to incrementally adjust parameters."""
    def __init__(self, hidden_size, dropout):
        """Creates an instance of the Residual_block class"""
        # initialise instance of nn.Module class
        super().__init__()

        # create two linear layers
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # ReLU activation
        self.relu = nn.ReLU()

        # introduce dropout
        self.dropout = nn.Dropout(dropout)

        # create two layers to normalise activations to mean = 0 and std. dev. = 1
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        """Propagates forwards through the residual block."""
        # store original x-value(s) for use later
        x_original = x
        
        # normalise data and propagate through first linear layer
        out = self.ln1(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # normalise data and propagate through second linear layer
        out = self.ln2(out)
        out = self.linear2(out)
        out = self.dropout(out)
        
        # residual connection
        out = out + x_original
        out = self.relu(out)
        return out

# ResNet class
class ResNet(Neural_net):
    """Specifies the architecture for a ResNet (residual network) neural network."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, dropout):
        """"""
        # initialise instance of Neural_net class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.dropout = dropout

        # store label
        self.label = "ResNet"

        # create input layer for lifting operation
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        
        # stack residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                Residual_block(self.hidden_size, dropout = dropout)
                for _ in range(self.no_of_layers - 2)
            ]
        )
        
        # create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # store network as a Sequential object
        self.net = nn.Sequential(
            self.input_layer,
            self.residual_blocks,
            self.output_layer
        )

        # print the number of parameters
        self.print_params()

# U_Net class
class U_Net(Neural_net):
    """Specifies the architecture for a U-Net neural network."""
    def __init__(self, no_of_layers, input_size, hidden_size, output_size, dropout):
        """"""
        # initialise instance of Neural_net class
        super().__init__(no_of_layers, input_size, hidden_size, output_size)

        # store input variables
        self.dropout = dropout

        # store label
        self.label = "U-Net"
        
        # create empty list of encoder layers
        self.encoder = nn.ModuleList()
        
        # loop for each layer
        for index in range(int(no_of_layers / 2) - 1):

            # append a layer to the encoder
            self.encoder.append(nn.Sequential(
                nn.Linear(input_size if index == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_size)
            ))
        
        # create bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # create empty list of decoder layers
        self.decoder = nn.ModuleList()

        # loop for each layer
        for index in range(int(no_of_layers / 2) - 1):

            # append a layer to the decoder
            self.decoder.append(nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_size)
            ))
        
        # create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # print number of parameters
        self.print_params()
    
    def forward(self, x):
        """Calculates the network forward pass with encoder-decoder and skip connections."""
        # create list to store skip connections and store original input value(s)
        skip_connections = []
        current = x
        
        # loop for each encoder layer
        for encoder_layer in self.encoder:

            # solve forward thrugh encoder layer and store skip connections
            current = encoder_layer(current)
            skip_connections.append(current)
        
        # solve forward through bottleneck
        current = self.bottleneck(current)

        # loop for each decoder layer
        for index, decoder_layer in enumerate(self.decoder):

            # utilise skip connections in reverse and solve forward through decoder layer
            skip = skip_connections[-(index + 1)]
            current = torch.cat([current, skip], dim = -1)
            current = decoder_layer(current)
        
        # return output
        out = self.output_layer(current)
        return out
