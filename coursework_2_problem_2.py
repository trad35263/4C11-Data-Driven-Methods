# import modules
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import perf_counter as timer
import datetime
import h5py

# import utils
import utils

# Inputs class
class Inputs:
    """Stores input parameters for the program."""
    # file paths for training/test data
    data_folder = "../Coursework 2/Coursework2/Coursework2_Problem_2/"
    training_path = "Darcy_2D_data_train.mat"
    test_path = "Darcy_2D_data_test.mat"

    # network hyperparameters
    no_of_layers = 3
    channel_width = 16

    # training variables
    no_of_epochs = 50
    batch_size = 20
    learning_rate = 5e-3
    weight_decay = 1e-4
    step_size = 200
    gamma = 0.5

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
    
# Define normaliser, pointwise gaussian
class UnitGaussianNormaliser(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormaliser, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

# Define network  
class CNN(nn.Module):
    """Convolutional Neural Network (CNN) model."""
    def __init__(self, base_channels = Inputs.channel_width):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            # encoder
            nn.Conv2d(1, base_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size = 3, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size = 3, padding=4, dilation = 4),
            nn.ReLU(),
            # decoder
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size = 3, padding = 4, dilation = 4),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size = 3, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, kernel_size = 3, padding = 1),
        )

    def forward(self, x):
        """Propagates through the neural network and returns the output."""
        # pass x through neural net and return output
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.squeeze(1)
        return out

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

    # normalise data
    a_train = a_normaliser.encode(a_train)
    a_test = a_normaliser.encode(a_test)

    print("Data shapes:")
    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    # create training data loader
    training_set = Data.TensorDataset(a_train, u_train)
    training_loader = Data.DataLoader(training_set, Inputs.batch_size, shuffle=True)

    # create neural network
    net = CNN()

    # print user feedback
    no_of_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {utils.Colours.GREEN}{no_of_parameters}{utils.Colours.END}")

    # define loss function
    loss_function = LpLoss()
    optimiser = torch.optim.Adam(
        net.parameters(), lr = Inputs.learning_rate, weight_decay = Inputs.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size = Inputs.step_size, gamma = Inputs.gamma
    )

    # print user feedback
    print(
        f"Training CNN for {utils.Colours.GREEN}{Inputs.no_of_epochs}{utils.Colours.END} "
        "epochs!"
    )
    t1 = timer()
    
    # create empty lists to store training and test loss
    training_loss_list = []
    test_loss_list = []
    x = []

    # loop for each epoch
    for epoch in range(Inputs.no_of_epochs):

        # set neural net to train and training loss to zero
        net.train(True)
        training_loss = 0

        # loop for each entry in the training data loader
        for i, data in enumerate(training_loader):

            # propagate forward through the neural network and calculate loss
            input, target = data
            output = net(input) # Forward
            output = u_normaliser.decode(output)
            loss = loss_function(output, target) # Calculate loss

            optimiser.zero_grad() # Clear gradients
            loss.backward() # Backward
            optimiser.step() # Update parameters
            scheduler.step() # Update learning rate

            training_loss += loss.item()    

        # Test
        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normaliser.decode(test_output)
            test_loss = loss_function(test_output, u_test).item()

        # every 10th epoch
        if epoch % 1 == 0:

            # print user feedback
            t2 = timer()
            print(
                f"epoch: {epoch}, "
                f"train loss: {training_loss / len(training_loader):.4g}, "
                f"test loss: {test_loss:.4g} "
                f"Time taken: {t2 - t1:.4g} s"
            )

        training_loss_list.append(training_loss / len(training_loader))
        test_loss_list.append(test_loss)
        x.append(epoch)

    t2 = timer()
    print(f"Traing time: {t2 - t1:.4g}")
    print(f"Train loss:{training_loss / len(training_loader)}")
    print(f"Test loss:{test_loss}")
    
    ############################# Plot #############################
    plt.figure(1)
    plt.plot(x, training_loss_list, label='Train loss')
    plt.plot(x, test_loss_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.legend()
    plt.grid()

# upon script execution
if __name__ == '__main__':

    # run main
    main()

    # show all plots
    plt.show()
