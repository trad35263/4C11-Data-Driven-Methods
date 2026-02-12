# import modules
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py

class Inputs:
    """Stores code inputs."""
    # file path of folder contain material data
    data_folder = "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/4C11 Data-Driven Methods/Project_1/Problem_1_student/Data"

    # batch size
    batch_size = 20

    # number of training datapoints
    ntrain = 600

    # number of training epochs
    no_of_epochs = 100

# Lossfunc class
class Lossfunc(nn.Module):
    """Mean squared error loss."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        return self.mse(prediction, target)

# MatRead class
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
    def __init__(self, data, eps=1e-8):
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

    # uncomment to plot
    """stress0 = stress[0]
    strain0 = strain[0]
    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot(
            [x[i] for x in stress0],
            [x[i] for x in strain0],
            label = f"{i}"
        )
    ax.legend()
    ax.grid()
    plt.show()"""

    # specify the training data
    ntrain = Inputs.ntrain
    train_strain = strain[:ntrain]
    train_stress = stress[:ntrain]

    # specify the test data
    test_strain = strain[len(strain) - Inputs.ntrain:]
    test_stress = stress[len(strain) - Inputs.ntrain:]

    # normalise strain data
    strain_normaliser = DataNormaliser(train_strain)
    train_strain_encode = strain_normaliser.transform(train_strain)
    test_strain_encode = strain_normaliser.transform(test_strain)

    # normalise stress data
    stress_normaliser = DataNormaliser(train_stress)
    train_stress_encode = stress_normaliser.transform(train_stress)
    test_stress_encode = stress_normaliser.transform(test_stress)

    # store number of components and timestep information separately for convenience
    ndim = strain.shape[2]
    nstep = strain.shape[1]
    dt = 1 / (nstep - 1)

    # create data loader for training data
    batch_size = Inputs.batch_size
    train_set = Data.TensorDataset(train_strain, train_stress)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle = True)

    # create data loader for test data
    test_set = Data.TensorDataset(test_strain, test_stress)
    test_loader = Data.DataLoader(test_set, batch_size, shuffle = True)

    # create neural network
    net = Const_Net()

    # calculate the number of training parameters
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params}")

    # define loss function
    loss_function = Lossfunc()

    """loss = loss_function(predicted_stress, true_stress)
    loss.backward()"""

    # define optimiser
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr = 1e-3,
        weight_decay = 0.0   # set >0 if you want L2 regularisation
    )

    # define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode = "min",
        factor = 0.5,
        patience = 10,
        verbose = True
    )

    # define number of training epochs
    no_of_epochs = Inputs.no_of_epochs
    print(f"Start training for {no_of_epochs} epochs...")

    # create empty lists to store loss convergence data
    loss_train_list = []
    loss_test_list = []

    # loop for each epoch
    for epoch in range(no_of_epochs):

        # train neural network
        net.train(True)
        trainloss = 0

        # loop through training data
        for index, (input, target) in enumerate(train_loader):

            # define forward neural network
            output_encode = net(input)

            # decode output
            output = strain_normaliser.inverse_transform(output_encode)

            # calculate loss
            loss = loss_function.forward(output, target)

                                # Clear gradients
                                # Backward
                                # Update parameters
                                # Update learning rate
            # update your train loss here
            trainloss += loss

        # Compute your test loss below
        net.eval()
        with torch.no_grad():

            testloss = 0

            # loop through training data
            for index, (input, target) in enumerate(test_loader):

                # define forward neural network
                output_encode = net(input)

                # decode output
                output = strain_normaliser.inverse_transform(output_encode)

                # calculate loss
                loss = loss_function.forward(output, target)

                                    # Clear gradients
                                    # Backward
                                    # Update parameters
                                    # Update learning rate
                # update your train loss here
                trainloss += loss

        # print train loss every 10 epochs
        if epoch % 10 == 0:

            print(
                f"epoch: {epoch}, train loss: {trainloss / len(train_loader)}, "
                f"test loss: {testloss}"
            )

        # save loss
        loss_train_list.append(trainloss / len(train_loader))
        loss_test_list.append(testloss)

    print(f"Train loss: {trainloss / len(train_loader)}")
    print(f"Test loss: {testloss}")

# runs on script execution
if __name__ == "__main__":
    """"""
    # run main
    main()

######################### Data processing #############################
# Read data from .mat file
# Define your data path here




############################# Define and train network #############################



# Train network



############################# Plot your result below using Matplotlib #############################
plt.figure(1)
plt.title('Train and Test Losses')

plt.figure(2)
plt.title('Truth Stresses vs Approximate Stresses for Sample {}')
