# import modules
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy

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
    sigma_1 = 0.1
    sigma_2 = 0

    # neural net architectural parameters
    no_of_layers = 1

    # training variables
    no_of_epochs = 100            # recommended: 50k - 100k

# Define Neural Network
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
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

    # set default datatype and device
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")         # or 'cuda' if using GPU

    # load data
    path = Inputs.data_folder + "/" + Inputs.file_name
    data = scipy.io.loadmat(path)

    # create pyTorch tensors
    L_boundary = torch.tensor(data["L_boundary"], dtype = torch.float64)
    R_boundary = torch.tensor(data["R_boundary"], dtype = torch.float64)
    T_boundary = torch.tensor(data["T_boundary"], dtype = torch.float64)
    B_boundary = torch.tensor(data["B_boundary"], dtype = torch.float64)
    C_boundary = torch.tensor(data["C_boundary"], dtype = torch.float64)
    x_boundary = torch.tensor(data["Boundary"], dtype = torch.float64, requires_grad = True)

    # get ground-truth displacement solution
    displacements = torch.tensor(data["disp_data"], dtype = torch.float64)
    
    # connectivity matrix - for plotting the figure but not needed for PINN
    connectivity_matrix = torch.tensor(data["t"].astype(float), dtype = torch.float64)

    # get collocation points - including the boundary
    x_full = torch.tensor(data["p_full"], dtype = torch.float64, requires_grad = True)

    # collocation points - excluding the boundary
    x = torch.tensor(data["p"], dtype = torch.float64, requires_grad = True)

    # choose 50 fixed points from the truth solution, used for part (e)
    random_index = torch.randint(0, len(x_full), (50,))
    displacements_fixed = displacements[random_index,:]

    # choose neural network layers
    displacement_net_layers = [2, 300, 300, 2] # Architecture of displacement net - you may change as you wish
    stress_net_layers = [2, 400, 400, 3] # Architecture of stress net - you may change as you wish
    
    # choose hyperbolic tangent as activation function and create neural nets
    displacement_net =  DenseNet(displacement_net_layers, nn.Tanh)
    stress_net = DenseNet(stress_net_layers, nn.Tanh)

    # calculate stiffness matrix via Hooke's law for plane stress
    E = Inputs.youngs_modulus
    nu = Inputs.poisson_ratio
    stiffness_matrix = (
        E / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    )
    stiffness_matrix = stiffness_matrix.unsqueeze(0)

    # define loss function - try pyTorch built-in MSE loss
    loss_func = torch.nn.MSELoss()

    # broadcast stiffness matrix for batch multiplication later
    stiffness_matrix_boundary = stiffness_matrix
    stiffness_matrix = torch.broadcast_to(stiffness_matrix, (len(x), 3, 3))
    stiffness_matrix_boundary = (
        torch.broadcast_to(stiffness_matrix_boundary, (len(x_boundary), 3, 3))
    )

    # get list of parameters and print user feedback
    parameters = list(stress_net.parameters()) + list(displacement_net.parameters())
    print(f"Total number of parameters: {utils.Colours.GREEN}{len(parameters)}{utils.Colours.END}")

    # define optimiser and scheduler - try standard implementation
    optimiser = torch.optim.Adam(parameters, lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = 100, gamma = 0.5)

    # loop for each epoch
    for epoch in range(Inputs.no_of_epochs):

        # reset all gradient tensors of the model parameters to zero
        optimiser.zero_grad()

        # get stress and displacement from respective neural nets
        stress = stress_net(x)
        displacement = displacement_net(x)

        # package results in object-oriented fashion
        bulk = Stress_strain(x, stress, displacement)

        # define augmented stress
        bulk.stress_augmented = torch.bmm(stiffness_matrix, bulk.e).squeeze(2)

        # define constitutive loss - forcing the augment stress to be equal to the neural network stress
        bulk.loss_constitutive = loss_func(bulk.stress_augmented, stress)

        # find displacement and stress at the boundaries
        displacement_boundary = displacement_net(x_boundary)
        stress_boundary = stress_net(x_boundary)

        # repeat for boundary nodes
        boundary = Stress_strain(x_boundary, stress_boundary, displacement_boundary)

        # define augmented stress
        boundary.stress_augmented = torch.bmm(stiffness_matrix_boundary, boundary.e).squeeze(2)

        # force the augment stress to agree with the NN stress at the boundary
        boundary.loss_constitutive = loss_func(boundary.stress_augmented, stress_boundary)

        # extract stress components
        stress_11 = stress[:, 0]
        stress_22 = stress[:, 1]
        stress_12 = stress[:, 2]

        # calculate derivatives of each stress component
        dstress_11_dx = torch.autograd.grad(
            stress_11, x, grad_outputs = torch.ones_like(stress_11), create_graph = True
        )[0]
        dstress_22_dx = torch.autograd.grad(
            stress_22, x, grad_outputs = torch.ones_like(stress_11), create_graph = True
        )[0]
        dstress_12_dx = torch.autograd.grad(
            stress_12, x, grad_outputs = torch.ones_like(stress_11), create_graph = True
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
        u_L = displacement_net(L_boundary)
        u_B = displacement_net(B_boundary)

        # stress boundary conditions
        stress_R = stress_net(R_boundary)
        stress_T = stress_net(T_boundary)
        stress_C = stress_net(C_boundary)

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
                stress_C[:, 0] * C_boundary[:, 0] + stress_C[:, 2] * C_boundary[:, 1],
                torch.zeros_like(stress_C[:, 0])
            ) + loss_func(
                stress_C[:, 2] * C_boundary[:, 0] + stress_C[:, 1] * C_boundary[:, 1],
                torch.zeros_like(stress_C[:, 0])
            )
        )

        # define loss function
        loss = (
            loss_equilibrium_x1 + loss_equilibrium_x2 + bulk.loss_constitutive + loss_BC_L
            + loss_BC_B + loss_BC_R + loss_BC_T
            + loss_BC_C + boundary.loss_constitutive
        )

        # ======= uncomment below for part (e) =======================
        # data_loss_fix
        #x_fix = x_full[rand_index, :]
        #u_fix = disp_net(x_fix)
        #loss_fix = loss_func(u_fix,displacements_fixed)
        #loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc + 100*loss_fix

        # calculate gradients of the loss with respect to model parameters
        loss.backward()

        # step optimiser and scheduler
        optimiser.step()
        scheduler.step()

        # print user feedback
        print(
            f"Epoch no.: {utils.Colours.CYAN}{epoch}{utils.Colours.END}\n"
            f"Training loss: {utils.Colours.GREEN}{loss:.4g}{utils.Colours.END}"
        )

    # plot the stress
    import matplotlib.tri as mtri

    stiff = E / (1 - nu ** 2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    stiff = stiff.unsqueeze(0)

    stiff_bc = stiff
    stiff_full = stiff
    stiff = torch.broadcast_to(stiff, (len(x), 3, 3))

    stiff_bc = torch.broadcast_to(stiff_bc, (len(x_boundary), 3, 3))
    stiff_full = torch.broadcast_to(stiff_full, (len(x_full), 3, 3))

    u_full = displacement_net(x_full)
    stress_full = stress_net(x_full)

    xx = x_full[:,0].detach().numpy()
    yy = x_full[:,1].detach().numpy()
    sig11 = stress_full[:,1].detach().numpy()

    connect =(connectivity_matrix -1).detach().numpy()

    triang = mtri.Triangulation(xx, yy, connect)

    u_11 = u_full[:,0].detach().numpy()

    u = u_full[:, 0]
    v = u_full[:, 1]

    dudx = torch.autograd.grad(u, x_full, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dvdx = torch.autograd.grad(v, x_full, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    e_11 = dudx[:, 0].unsqueeze(1)
    e_22 = dvdx[:, 1].unsqueeze(1)
    e_12 = 0.5 * (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)

    e = torch.cat((e_11, e_22, e_12), 1)
    e = e.unsqueeze(2)

    sigma = torch.bmm(stiff_full, e).squeeze(2)

    # configure plot
    plt.figure(2)
    plt.clf()
    plt.tricontourf(triang,sigma[:,0].detach().numpy())
    plt.colorbar()

# upon script execution
if __name__ == "__main__":

    # run main
    main()

    # show all plots
    plt.show()


############################# Data processing #############################

# We will use two neural networks for the problem:
# NN1: to map the coordinates [x,y] to displacement u
# NN2: to map the coordinates [x,y] to the stresses [sigma_11, sigma_22, sigma_12]
# What we will do later is to first compute strain by differentiate the output of NN1
# And then we compute a augment stress using Hook's law to find an augmented stress sigma_a
# And we will require the output of NN2 to match sigma_a  - we shall do this by adding a term in the loss function
# This will help us to avoid differentiating NN1 twice (why?)
# As it is well known that PINN suffers from higher order derivatives

# Plot the stress

