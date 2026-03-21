# import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
import torch.nn as nn
import scipy
from time import perf_counter as timer

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

    # material properties
    youngs_modulus = 10             # Young's modulus (N/m^2)
    poisson_ratio = 0.3             # Poisson ratio (-)
    thickness = 1                   # Plate thickness (m)

    # applied stress
    sigma_1 = 0.1                   # N/m^2
    sigma_2 = 0                     # N/m^2

    # neural net architectural parameters
    no_of_layers = 3
    input_size = 2

    # construct stress net layer configuration
    stress_hidden_layer_size = 300
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
    no_of_epochs = 200            # recommended: 50k - 100k
    step_size = 100
    gamma = 0.5

    # plotting parameters
    figsize = (9, 5)
    fontsize = 12
    titlesize = 14

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

        # calculate stiffness matrix via Hooke's law for plane stress
        E = Inputs.youngs_modulus
        nu = Inputs.poisson_ratio
        stiffness_matrix = (
            E / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
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

    def plot_stress(self, displacement_net, stress_net):
        """Plots the stress field over the full domain."""
        # prepare labels for each stress component
        component_labels = {0: r"$\sigma_{11}$", 1: r"$\sigma_{22}$", 2: r"$\sigma_{12}$"}

        # get displacements and stresses from respective neural networks
        u_full = displacement_net(self.x_full)
        stress_full = stress_net(self.x_full)

        # build triangulation
        xx = self.x_full[:, 0].detach().numpy()
        yy = self.x_full[:, 1].detach().numpy()
        connect = (self.connectivity_matrix - 1).detach().numpy()
        triang = mtri.Triangulation(xx, yy, connect)

        # compute strains via autograd
        u, v = u_full[:, 0], u_full[:, 1]
        du_dx = torch.autograd.grad(
            u, self.x_full, grad_outputs = torch.ones_like(u), create_graph = True
        )[0]
        dv_dx = torch.autograd.grad(
            v, self.x_full, grad_outputs = torch.ones_like(v), create_graph = True
        )[0]

        # calculate individual strain components
        e_11 = du_dx[:, 0].unsqueeze(1)
        e_22 = dv_dx[:, 1].unsqueeze(1)
        e_12 = (0.5 * (du_dx[:, 1] + dv_dx[:, 0])).unsqueeze(1)

        # calculat strain vector
        e = torch.cat((e_11, e_22, e_12), 1).unsqueeze(2)

        # compute stress from constitutive relation
        sigma = torch.bmm(self.stiffness_matrix_full, e).squeeze(2)

        # loop for each stress component
        for index in range(3):

            # create plot
            fig, ax = plt.subplots(figsize = Inputs.figsize)
            plot = ax.tricontourf(triang, sigma[:, index].detach().numpy())
            plt.colorbar(plot, ax = ax)

            # configure plot
            ax.set_title(
                f"Stress field: {component_labels[index]}", fontsize = Inputs.titlesize
            )
            ax.set_xlabel("x", fontsize = Inputs.fontsize)
            ax.set_ylabel("y", fontsize = Inputs.fontsize)
            ax.set_aspect("equal")

            # create plot
            fig, ax = plt.subplots(figsize = Inputs.figsize)
            plot = ax.tricontourf(
                triang,sigma[:, index].detach().numpy() - stress_full[:, index].detach().numpy()
            )
            plt.colorbar(plot, ax = ax)

            # configure plot
            ax.set_title(
                f"Stress field: {component_labels[index]} RESIDUALS", fontsize = Inputs.titlesize
            )
            ax.set_xlabel("x", fontsize = Inputs.fontsize)
            ax.set_ylabel("y", fontsize = Inputs.fontsize)
            ax.set_aspect("equal")

    def plot_fix_accuracy(self, displacement_net):
        """Plots predicted vs ground-truth displacements at the 50 fixed points."""

        x_fix = self.x_full[self.random_index, :]
        u_fix = displacement_net(x_fix).detach().numpy()
        u_true = self.displacements_fixed.detach().numpy()

        fig, axes = plt.subplots(1, 2, figsize=Inputs.figsize)

        for i, (ax, label) in enumerate(zip(axes, ["u", "v"])):
            ax.scatter(u_true[:, i], u_fix[:, i], s=10)

            # plot y=x line for perfect agreement
            lims = [u_true[:, i].min(), u_true[:, i].max()]
            ax.plot(lims, lims, 'r--', label="Perfect agreement")

            ax.set_xlabel(f"{label} true", fontsize=Inputs.fontsize)
            ax.set_ylabel(f"{label} predicted", fontsize=Inputs.fontsize)
            ax.set_title(f"Fixed point accuracy: {label}", fontsize=Inputs.titlesize)
            ax.legend(fontsize=Inputs.fontsize)
            ax.grid()
            ax.set_aspect("equal")

# Define Neural Network
class DenseNet(nn.Module):
    """Creates an instance of the neural network."""
    def __init__(self, layers, nonlinearity):
        """Creates an instance of the DenseNet class."""
        # initialise parent class
        super(DenseNet, self).__init__()

        # store layers as a class attribute
        self.no_of_layers = len(layers) - 1
        self.layers = nn.ModuleList()

        # loop for each layer
        for index in range(self.no_of_layers):

            # append layer with appropriate input and output size
            self.layers.append(nn.Linear(layers[index], layers[index + 1]))

            # for all but the final layer
            if index != self.no_of_layers - 1:

                # append non-linearity
                self.layers.append(nonlinearity())

        # initialise empty list for storing training loss
        self.training_loss = []

    def forward(self, x):
        """Propagates forwards through the neural network and calculates the output."""
        # loop for each network layer
        for layer in self.layers:

            # calculate output of layer
            x = layer(x)

        # return neural network output
        return x

    def plot_convergence(self):
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
            u, x, grad_outputs = torch.ones_like(u), create_graph = True
        )[0]
        dv_dx = torch.autograd.grad(
            v, x, grad_outputs = torch.ones_like(u), create_graph = True
        )[0]

        # define strains in the orthogonal and shear directions
        e_11 = du_dx[:, 0].unsqueeze(1)
        e_22 = dv_dx[:, 1].unsqueeze(1)
        e_12 = (0.5 * (du_dx[:, 1] + dv_dx[:, 0])).unsqueeze(1)

        # define strain vector by combining components and add a trailing dimension
        e = torch.cat((e_11, e_22, e_12), 1)
        self.e = e.unsqueeze(2)

# main function
def main():

    # create instance of Data class
    data = Data()
    
    # choose hyperbolic tangent as activation function and create neural nets
    displacement_net =  DenseNet(Inputs.displacement_net_layers, nn.Tanh)
    stress_net = DenseNet(Inputs.stress_net_layers, nn.Tanh)

    # define loss function - try pyTorch built-in MSE loss
    loss_func = torch.nn.MSELoss()

    # get list of parameters and print user feedback
    parameters = list(stress_net.parameters()) + list(displacement_net.parameters())
    no_of_parameters = sum(p.numel() for p in parameters)
    print(f"Total number of parameters: {utils.Colours.GREEN}{no_of_parameters}{utils.Colours.END}")

    # define optimiser and scheduler - try standard implementation
    optimiser = torch.optim.Adam(parameters, lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size = Inputs.step_size, gamma = Inputs.gamma
    )

    # print user feedback
    t1 = timer()
    print(
        f"Training model for {utils.Colours.GREEN}{Inputs.no_of_epochs}{utils.Colours.END} epochs!"
    )

    # loop for each epoch
    for epoch in range(Inputs.no_of_epochs):

        # reset all gradient tensors of the model parameters to zero
        optimiser.zero_grad()

        # get stress and displacement from respective neural nets
        stress = stress_net(data.x)
        displacement = displacement_net(data.x)

        # package results in object-oriented fashion
        bulk = Stress_strain(data.x, stress, displacement)

        # define augmented stress
        bulk.stress_augmented = torch.bmm(data.stiffness_matrix, bulk.e).squeeze(2)

        # define constitutive loss - forcing the augmented stress to be equal to the neural net stress
        bulk.loss_constitutive = loss_func(bulk.stress_augmented, stress)

        # find displacement and stress at the boundaries
        displacement_boundary = displacement_net(data.boundary["x"])
        stress_boundary = stress_net(data.boundary["x"])

        # repeat for boundary nodes
        boundary = Stress_strain(data.boundary["x"], stress_boundary, displacement_boundary)

        # define augmented stress
        boundary.stress_augmented = torch.bmm(data.stiffness_matrix_boundary, boundary.e).squeeze(2)

        # force the augment stress to agree with the neural net stress at the boundary
        boundary.loss_constitutive = loss_func(boundary.stress_augmented, stress_boundary)

        # extract stress components
        stress_11 = stress[:, 0]
        stress_22 = stress[:, 1]
        stress_12 = stress[:, 2]

        # calculate derivatives of each stress component
        dstress_11_dx = torch.autograd.grad(
            stress_11, data.x, grad_outputs = torch.ones_like(stress_11), create_graph = True
        )[0]
        dstress_22_dx = torch.autograd.grad(
            stress_22, data.x, grad_outputs = torch.ones_like(stress_11), create_graph = True
        )[0]
        dstress_12_dx = torch.autograd.grad(
            stress_12, data.x, grad_outputs = torch.ones_like(stress_11), create_graph = True
        )[0]

        # enforce Newton's second law for static equilibrium
        equilibrium_x1 = dstress_11_dx[:, 0] + dstress_12_dx[:, 1]
        equilibrium_x2 = dstress_22_dx[:, 0] + dstress_12_dx[:, 1]

        # zero body forces
        force_x1 = torch.zeros_like(equilibrium_x1)
        force_x2 = torch.zeros_like(equilibrium_x2)

        # calculate losses assocated with stress equilibrium equations
        loss_equilibrium_x1 = loss_func(equilibrium_x1, force_x1)
        loss_equilibrium_x2 = loss_func(equilibrium_x2, force_x2)

        # specify the boundary condition
        tau_R = 0
        tau_T = 0

        # displacement boundary conditions
        u_L = displacement_net(data.boundary["left"])
        u_B = displacement_net(data.boundary["bottom"])

        # stress boundary conditions
        stress_R = stress_net(data.boundary["right"])
        stress_T = stress_net(data.boundary["top"])
        stress_C = stress_net(data.boundary["circle"])

        # apply symmetry boundary condition on the left edge
        loss_BC_L = loss_func(u_L[:,0], torch.zeros_like(u_L[:,0]))

        # apply symmetry boundary condition on the bottom edge
        loss_BC_B = loss_func(u_B[:, 1], torch.zeros_like(u_B[:, 1]))

        # apply force boundary condition on the right edge
        loss_BC_R = (
            loss_func(stress_R[:, 0], tau_R * torch.ones_like(stress_R[:, 0]))
            + loss_func(stress_R[:, 2], torch.zeros_like(stress_R[:, 2]))
        )

        # apply force boundary condition on the top edge
        loss_BC_T = (
            loss_func(stress_T[:, 1], tau_T * torch.ones_like(stress_T[:, 1]))
            + loss_func(stress_T[:, 2], torch.zeros_like(stress_T[:, 2]))
        )

        # traction free on circle
        loss_BC_C = (
            loss_func(
                stress_C[:, 0] * data.boundary["circle"][:, 0] + stress_C[:, 2] * data.boundary["circle"][:, 1],
                torch.zeros_like(stress_C[:, 0])
            ) + loss_func(
                stress_C[:, 2] * data.boundary["circle"][:, 0] + stress_C[:, 1] * data.boundary["circle"][:, 1],
                torch.zeros_like(stress_C[:, 0])
            )
        )

        # calculate training loss
        training_loss = (
            loss_equilibrium_x1 + loss_equilibrium_x2 + bulk.loss_constitutive + loss_BC_L
            + loss_BC_B + loss_BC_R + loss_BC_T
            + loss_BC_C + boundary.loss_constitutive
        )

        # ======= uncomment below for part (e) =======================
        # data_loss_fix
        """x_fix = data.x_full[data.random_index, :]
        u_fix = displacement_net(x_fix)
        loss_fix = loss_func(u_fix, data.displacements_fixed)
        training_loss = (
            loss_equilibrium_x1 + loss_equilibrium_x2 + bulk.loss_constitutive + loss_BC_L
            + loss_BC_B + loss_BC_R + loss_BC_T
            + loss_BC_C + boundary.loss_constitutive + 100 * loss_fix
        )"""

        # calculate gradients of the loss with respect to model parameters
        training_loss.backward()

        # step optimiser and scheduler
        optimiser.step()
        scheduler.step()

        # store training loss in neural network
        stress_net.training_loss.append(float(training_loss))

        if epoch % 10 == 0:

            t2 = timer()

            # print user feedback
            print(
                f"Epoch no.: {utils.Colours.CYAN}{epoch}{utils.Colours.END}\n"
                f"Training loss: {utils.Colours.GREEN}{training_loss:.4g}{utils.Colours.END}\n"
                f"Time taken: {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s"
            )

    stress_net.no_of_epochs = Inputs.no_of_epochs
    stress_net.plot_convergence()
    data.plot_stress(displacement_net, stress_net)
    data.plot_fix_accuracy(displacement_net)

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
def save_figure(fig, ax, name):

    # draw canvas to render dimensions and get legend
    fig.canvas.draw()
    legend = ax.get_legend()

    if legend == None:

        legend_width = 0

    else:

        legend_width = legend.get_window_extent().width / fig.dpi

    total_width = Inputs.left_margin + Inputs.ax_width + legend_width + Inputs.right_margin

    # set total figure width and height
    fig.set_size_inches(
        total_width,
        Inputs.ax_height
    )

    # explicitly pin axes position in inch-accurate fractions
    ax.set_position([
        Inputs.left_margin / total_width,
        Inputs.bottom_margin / Inputs.ax_height,
        Inputs.ax_width / total_width,
        Inputs.plot_height / Inputs.ax_height
    ])

    # save plot
    fig.savefig(name, dpi = Inputs.dpi, bbox_inches = "tight")

# upon script execution
if __name__ == "__main__":

    # set default datatype and device
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")         # or 'cuda' if using GPU

    # run main
    main()

    # show all plots
    plt.show()
