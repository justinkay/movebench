"""
deepSSF_model.py
==============

Description:
    This script contains the deepSSF model classes and the dictionaries 
    used in the deepSSF_train script.

Authors:
    Scott Forrest (scottwforrest@gmail.com)
    Dan Pagendam  (dan.pagendam@data61.csiro.au)

Date:
    2025-02-02
    
Usage:
    Run the script as a module, such as in a Jupyter notebook or Python console:
        >>> import deepSSF_model
        >>> deepSSF_model.ConvJointModel(args)
    
    Or execute from the command line:
        $ python deepSSF_model.py [optional arguments]
"""

# Standard library imports
import torch
import numpy as np

# # Third-party imports
from torch import nn

# Set the device to be used (GPU or CPU)
# if torch.cuda.is_available():
#     device = "cuda"
# # elif torch.backends.mps.is_available():  # For Mac M1/M2/M3
# #     device = "mps"
# else:
#     device = "cpu"
    
# print(f"Using {device} device")

# if torch.backends.mps.is_available():
#     # Set default tensor type for PyTorch
#     torch.set_default_dtype(torch.float32)
#     print('Set default tensor type to float32')


"""
## Convolutional block for the habitat selection subnetwork

This block is a convolutional layer that takes in the spatial covariates 
(including the layers created from the scalar values such as time), 
goes through a series of convolution operations amd ReLU activation functions and 
outputs a feature map, which is the habitat selection probability surface.

"""
class Conv2d_block_spatial(nn.Module):
    def __init__(self, params):
        super(Conv2d_block_spatial, self).__init__()

        # define the parameters
        self.batch_size = params.batch_size
        self.input_channels = params.input_channels
        self.output_channels = params.output_channels
        self.kernel_size = params.kernel_size
        self.stride = params.stride
        self.padding = params.padding
        self.image_dim = params.image_dim
        self.device = params.device

        # define the layers - nn.Sequential allows for the definition of layers in a sequential manner
        self.conv2d = nn.Sequential(
        # convolutional layer 1
        nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        # ReLU activation function
        nn.ReLU(),
        # convolutional layer 2
        nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        # ReLU activation function
        nn.ReLU(),
        # convolutional layer 3
        nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        # ReLU activation function
        nn.ReLU(),
        # convolutional layer 4, which outputs a single layer, which is the habitat selection map
        nn.Conv2d(in_channels=self.output_channels, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        )

    # define the forward pass of the model, i.e. how the data flows through the model
    def forward(self, x):

        # self.conv2d(x) passes the input through the convolutional layers, and the squeeze function removes the channel dimension, resulting in a 2D tensor (habitat selection map)
        # print("Shape before squeeze:", self.conv2d(x).shape) # Debugging print
        conv2d_spatial = self.conv2d(x).squeeze(dim = 1)
    
        # normalise to sum to 1
        # print("Shape before logsumexp:", conv2d_spatial.shape) # Debugging print
        conv2d_spatial = conv2d_spatial - torch.logsumexp(conv2d_spatial, dim = (1, 2), keepdim = True)

        # output the habitat selection map
        return conv2d_spatial
    

""""
## Convolutional block for the movement subnetwork

This block is also convolutional layer, with the same inputs, 
but this block also has max pooling layers to reduce the spatial 
resolution of the feature maps whilst preserving the most prominent 
features in the feature maps, and outputs a 'flattened' feature map. 
A flattened feature map is a 1D tensor (a vector) that can be used as 
input to a fully connected layer.

"""
class Conv2d_block_toFC(nn.Module):
    def __init__(self, params):
        super(Conv2d_block_toFC, self).__init__()

        # define the parameters
        self.batch_size = params.batch_size
        self.input_channels = params.input_channels
        self.output_channels = params.output_channels
        self.kernel_size = params.kernel_size
        self.stride = params.stride
        self.kernel_size_mp = params.kernel_size_mp
        self.stride_mp = params.stride_mp
        self.padding = params.padding
        self.image_dim = params.image_dim
        self.device = params.device

        # define the layers - nn.Sequential allows for the definition of layers in a sequential manner
        self.conv2d = nn.Sequential(
            
        # convolutional layer 1
        nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        # ReLU activation function
        nn.ReLU(),
        # max pooling layer 1 (reduces the spatial dimensions of the data whilst retaining the most important features)
        nn.MaxPool2d(kernel_size=self.kernel_size_mp, stride=self.stride_mp),
        
        # convolutional layer 2
        nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        # ReLU activation function
        nn.ReLU(),
        # max pooling layer 2
        nn.MaxPool2d(kernel_size=self.kernel_size_mp, stride=self.stride_mp),
        
        # convolutional layer 3
        nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        # ReLU activation function
        nn.ReLU(),
        # max pooling layer 2
        nn.MaxPool2d(kernel_size=self.kernel_size_mp, stride=self.stride_mp),
        
        # flatten the data to pass through the fully connected layer
        nn.Flatten())

    def forward(self, x):

        # self.conv2d(x) passes the input through the convolutional layers, and outputs a 1D tensor
        return self.conv2d(x)


"""
## Fully connected block for the movement subnetwork

This block takes in the flattened feature map from the previous block, 
passes through several fully connected layers, which extracts information 
from the spatial covariates that is relevant for movement, and outputs the 
parameters that define the movement kernel.

"""
class FCN_block_all_movement(nn.Module):
    def __init__(self, params):
        super(FCN_block_all_movement, self).__init__()

        # define the parameters
        self.batch_size = params.batch_size
        self.dense_dim_in_all = params.dense_dim_in_all
        self.dense_dim_hidden = params.dense_dim_hidden
        self.image_dim = params.image_dim
        self.device = params.device
        self.num_movement_params = params.num_movement_params
        self.dropout = params.dropout

        # define the layers - nn.Sequential allows for the definition of layers in a sequential manner
        self.ffn = nn.Sequential(

            # fully connected layer 1 (the dense_dim_in_all is the number of input features, 
            # and should match the output of the Conv2d_block_toFC block).
            # the dense_dim_hidden is the number of neurons in the hidden layer, and doesn't need to be the same as the input features
            nn.Linear(self.dense_dim_in_all, self.dense_dim_hidden),
            # dropout layer (helps to reduce overfitting)
            nn.Dropout(self.dropout),
            # ReLU activation function
            nn.ReLU(),

            # fully connected layer 2
            # the number of input neurons should match the output from the previous layer
            nn.Linear(self.dense_dim_hidden, self.dense_dim_hidden),
            # dropout layer
            nn.Dropout(self.dropout),
            # ReLU activation function
            nn.ReLU(),

            # fully connected layer final
            # the number of input neurons should match the output from the previous layer, 
            # and the number of output neurons should match the number of movement parameters
            nn.Linear(self.dense_dim_hidden, self.num_movement_params)
        )

    def forward(self, x):

        # self.ffn(x) passes the input through the fully connected layers, and outputs a 1D tensor (vector of movement parameters)
        return self.ffn(x)


"""
## Block to convert the movement parameters to a probability distribution

### What the block does

This block is a bit longer and more involved, but there are no parameters in here 
that need to be learned (estimated). It is just a series of operations that are 
applied to the movement parameters to convert them to a probability distribution.

This block takes in the movement parameters and converts them to a probability distribution. 
This essentially just applies the appropriate density functions using the parameter 
values predicted by the movement blocks, which in our case is a finite mixture of 
Gamma distributions nad a finite mixture of von Mises distributions.

The formulation of predicting parameters and converting them to a movement kernel 
ensures that the movement kernel is very flexible, and can be any combination of 
distributions, which need not all be the same (e.g., a step length distribution may 
be combination of a Gamma and a log-normal distribution).

### Constraints

One constraint to ensure that we can perform backpropagation is that the entire forward pass, 
including the block below that produces the density functions, must be differentiable 
with respect to the parameters of the model. PyTorch’s torch.distributions module and 
its special functions (e.g., torch.special) provide differentiable implementations for 
many common distributions. Examples are the

- Gamma function for the (log) Gamma distribution, `torch.lgamma()`
- The modified Bessel function of the first kind of order 0 for the von Mises distribution, `torch.special.i0()`

Some of the movement parameters, such as the shape and scale of the Gamma distribution, 
must be positive. We therefore exponentiate them in this block to ensure that they are positive. 
This means that the model is actually learning the log of the shape and scale parameters. 
For the von Mises `mu` parameters however, they can be any value, so we do not need to 
exponentiate them. We could constrain them to be between -pi and pi, but this is not 
necessary as the von Mises distribution is periodic, so any value will be equivalent to 
another value that is within the range -pi to pi.

### Notes

To help with identifiability, it is possible to fix certain parameter values, 
such as the mu parameters in the mixture of von Mises distributions to pi and -pi for instance 
(one would then reduce the number of predicted parameters by the previous block, 
as these no longer need to be predicted).

We can also transform certain parameters such that they are being estimated in a 
similar range (analagous to standardising variables in linear regression). 
In our case we know that the scale parameter of one of the Gamma distributions is around 500. 
What we can then do after exponentiating is multiply the scale parameter by 500, 
so the model is learning the log of the scale parameter divided by 500. 
This will ensure that this parameter is in a similar range to the other parameters, 
and can help with convergence. To do this we:

Pull out the relevant parameters from the input tensor (output of previous block)
- `gamma_scale2 = torch.exp(x[:, 4]).unsqueeze(0).unsqueeze(0)`

Multiply the scale parameter by 500, so the model is learning the log of the scale parameter divided by 500
- `gamma_scale2 = gamma_scale2 * 500` 

"""
class Params_to_Grid_Block(nn.Module):
    def __init__(self, params):
        super(Params_to_Grid_Block, self).__init__()

        # define the parameters
        self.batch_size = params.batch_size
        self.image_dim = params.image_dim
        self.pixel_size = params.pixel_size

        # create distance and bearing layers
        # determine the distance of each pixel from the centre of the image
        self.center = self.image_dim // 2 
        y, x = np.indices((self.image_dim, self.image_dim))
        self.distance_layer = torch.from_numpy(np.sqrt((self.pixel_size*(x - self.center))**2 + (self.pixel_size*(y - self.center))**2)).float()

        # average distance from the centre to the perimeter of the pixel (accounting for longer distances at the corners)
        # self.distance_layer[self.center, self.center] = 0.56*self.pixel_size 

        # average distance from the centre to any point within the pixel
        # calculated as a double integral of sqrt(x^2 + y^2) dx dy over the area of the pixel
        self.distance_layer[self.center, self.center] = 0.3826*self.pixel_size 

        # determine the bearing of each pixel from the centre of the image
        self.bearing_layer = torch.from_numpy(np.arctan2(self.center - y, x - self.center)).float()
        self.device = params.device


    # Gamma densities (on the log-scale) for the mixture distribution
    def gamma_density(self, x, shape, scale):
        # Ensure all tensors are on the same device as x
        shape = shape.to(x.device)
        scale = scale.to(x.device)
        return -1*torch.lgamma(shape) -shape*torch.log(scale) + (shape - 1)*torch.log(x) - x/scale
        
        # to account for change of variables
        # return (-1*torch.lgamma(shape) -shape*torch.log(scale) + (shape - 1)*torch.log(x) - x/scale) - torch.log(x)

    # log von Mises densities (on the log-scale) for the mixture distribution
    def vonmises_density(self, x, kappa, vm_mu):
        # Ensure all tensors are on the same device as x
        kappa = kappa.to(x.device)
        vm_mu = vm_mu.to(x.device)
        return kappa*torch.cos(x - vm_mu) - 1*(np.log(2*torch.pi) + torch.log(torch.special.i0(kappa)))


    def forward(self, x, bearing):

        # parameters of the first mixture distribution
        # x are the outputs from the fully connected layers (vector of movement parameters)
        # we therefore need to extract the appropriate parameters 
        # the locations are not specific to any specific parameters, as long as any aren't extracted more than once 

        # Gamma distributions

        # pull out the parameters of the first gamma distribution and exponentiate them to ensure they are positive
        # the unsqueeze function adds a new dimension to the tensor
        # we do this twice to match the dimensions of the distance_layer, 
        # and then repeat the parameter value across a grid, such that the density can be calculated at every cell/pixel
        gamma_shape1 = torch.exp(x[:, 0]).unsqueeze(0).unsqueeze(0)
        gamma_shape1 = gamma_shape1.repeat(self.image_dim, self.image_dim, 1)
        # this just changes the order of the dimensions to match the distance_layer
        gamma_shape1 = gamma_shape1.permute(2, 0, 1)

        gamma_scale1 = torch.exp(x[:, 1]).unsqueeze(0).unsqueeze(0)
        gamma_scale1 = gamma_scale1.repeat(self.image_dim, self.image_dim, 1)
        gamma_scale1 = gamma_scale1.permute(2, 0, 1)

        gamma_weight1 = torch.exp(x[:, 2]).unsqueeze(0).unsqueeze(0)
        gamma_weight1 = gamma_weight1.repeat(self.image_dim, self.image_dim, 1)
        gamma_weight1 = gamma_weight1.permute(2, 0, 1)

        # parameters of the second mixture distribution
        gamma_shape2 = torch.exp(x[:, 3]).unsqueeze(0).unsqueeze(0)
        gamma_shape2 = gamma_shape2.repeat(self.image_dim, self.image_dim, 1)
        gamma_shape2 = gamma_shape2.permute(2, 0, 1)

        gamma_scale2 = torch.exp(x[:, 4]).unsqueeze(0).unsqueeze(0)
        gamma_scale2 = gamma_scale2 * 500 ### transform the scale parameter so it can be estimated near the same range as the other parameters
        gamma_scale2 = gamma_scale2.repeat(self.image_dim, self.image_dim, 1)
        gamma_scale2 = gamma_scale2.permute(2, 0, 1)

        gamma_weight2 = torch.exp(x[:, 5]).unsqueeze(0).unsqueeze(0)
        gamma_weight2 = gamma_weight2.repeat(self.image_dim, self.image_dim, 1)
        gamma_weight2 = gamma_weight2.permute(2, 0, 1)

        # Apply softmax to the mixture weights to ensure they sum to 1
        gamma_weights = torch.stack([gamma_weight1, gamma_weight2], dim=0)
        gamma_weights = torch.nn.functional.softmax(gamma_weights, dim=0)
        gamma_weight1 = gamma_weights[0]
        gamma_weight2 = gamma_weights[1]

        # calculation of Gamma densities
        gamma_density_layer1 = self.gamma_density(self.distance_layer, gamma_shape1, gamma_scale1).to(self.device)
        gamma_density_layer2 = self.gamma_density(self.distance_layer, gamma_shape2, gamma_scale2).to(self.device)

        # combining both densities to create a mixture distribution using logsumexp
        logsumexp_gamma_corr = torch.max(gamma_density_layer1, gamma_density_layer2)
        gamma_density_layer = logsumexp_gamma_corr + torch.log(gamma_weight1 * torch.exp(gamma_density_layer1 - logsumexp_gamma_corr) + 
                                                               gamma_weight2 * torch.exp(gamma_density_layer2 - logsumexp_gamma_corr))
        # print(torch.sum(gamma_density_layer))
        # print(torch.sum(torch.exp(gamma_density_layer)))


        ## Von Mises Distributions

        # calculate the new bearing from the turning angle
        # takes in the bearing from the previous step and adds the turning angle, which is estimated by the model
        # we do not exponentiate the von Mises mu parameters as we want to allow them to be negative
        bearing_new1 = x[:, 6] + bearing[:, 0]

        # the new bearing becomes the mean of the von Mises distribution
        vonmises_mu1 = bearing_new1.unsqueeze(0).unsqueeze(0)
        vonmises_mu1 = vonmises_mu1.repeat(self.image_dim, self.image_dim, 1)
        vonmises_mu1 = vonmises_mu1.permute(2, 0, 1)

        # parameters of the first von Mises distribution
        vonmises_kappa1 = torch.exp(x[:, 7]).unsqueeze(0).unsqueeze(0)
        vonmises_kappa1 = vonmises_kappa1.repeat(self.image_dim, self.image_dim, 1)
        vonmises_kappa1 = vonmises_kappa1.permute(2, 0, 1)

        vonmises_weight1 = torch.exp(x[:, 8]).unsqueeze(0).unsqueeze(0)
        vonmises_weight1 = vonmises_weight1.repeat(self.image_dim, self.image_dim, 1)
        vonmises_weight1 = vonmises_weight1.permute(2, 0, 1)

        # vm_mu and weight for the second von Mises distribution
        bearing_new2 = x[:, 9] + bearing[:, 0]

        vonmises_mu2 = bearing_new2.unsqueeze(0).unsqueeze(0)
        vonmises_mu2 = vonmises_mu2.repeat(self.image_dim, self.image_dim, 1)
        vonmises_mu2 = vonmises_mu2.permute(2, 0, 1)

        # parameters of the second von Mises distribution
        vonmises_kappa2 = torch.exp(x[:, 10]).unsqueeze(0).unsqueeze(0)
        vonmises_kappa2 = vonmises_kappa2.repeat(self.image_dim, self.image_dim, 1)
        vonmises_kappa2 = vonmises_kappa2.permute(2, 0, 1)

        vonmises_weight2 = torch.exp(x[:, 11]).unsqueeze(0).unsqueeze(0)
        vonmises_weight2 = vonmises_weight2.repeat(self.image_dim, self.image_dim, 1)
        vonmises_weight2 = vonmises_weight2.permute(2, 0, 1)

        # Apply softmax to the weights
        vonmises_weights = torch.stack([vonmises_weight1, vonmises_weight2], dim=0)
        vonmises_weights = torch.nn.functional.softmax(vonmises_weights, dim=0)
        vonmises_weight1 = vonmises_weights[0]
        vonmises_weight2 = vonmises_weights[1]

        # calculation of von Mises densities
        vonmises_density_layer1 = self.vonmises_density(self.bearing_layer, vonmises_kappa1, vonmises_mu1).to(self.device)
        vonmises_density_layer2 = self.vonmises_density(self.bearing_layer, vonmises_kappa2, vonmises_mu2).to(self.device)

        # combining both densities to create a mixture distribution using the logsumexp trick
        logsumexp_vm_corr = torch.max(vonmises_density_layer1, vonmises_density_layer2)
        vonmises_density_layer = logsumexp_vm_corr + torch.log(vonmises_weight1 * torch.exp(vonmises_density_layer1 - logsumexp_vm_corr) + vonmises_weight2 * torch.exp(vonmises_density_layer2 - logsumexp_vm_corr))
        # print(torch.sum(vonmises_density_layer))
        # print(torch.sum(torch.exp(vonmises_density_layer)))

        # combining the two distributions
        movement_grid = gamma_density_layer + vonmises_density_layer # Gamma and von Mises densities are on the log-scale

        # normalise (on the log-scale using the log-sum-exp trick) before combining with the habitat predictions
        movement_grid = movement_grid - torch.logsumexp(movement_grid, dim = (1, 2), keepdim = True)
        # print('Movement grid norm ', torch.sum(movement_grid))
        # print(torch.sum(torch.exp(movement_grid)))

        return movement_grid
    

"""
Same as above but accounting for the change of variables from polar to Cartesian coordinates.
"""
class Params_to_Grid_Block_ChV(nn.Module):
    def __init__(self, params):
        super(Params_to_Grid_Block_ChV, self).__init__()

        # define the parameters
        self.batch_size = params.batch_size
        self.image_dim = params.image_dim
        self.pixel_size = params.pixel_size

        # create distance and bearing layers
        # determine the distance of each pixel from the centre of the image
        self.center = self.image_dim // 2 
        y, x = np.indices((self.image_dim, self.image_dim))
        self.distance_layer = torch.from_numpy(np.sqrt((self.pixel_size*(x - self.center))**2 + (self.pixel_size*(y - self.center))**2)).float()

        # average distance from the centre to the perimeter of the pixel (accounting for longer distances at the corners)
        # self.distance_layer[self.center, self.center] = 0.56*self.pixel_size 

        # average distance from the centre to any point within the pixel
        # calculated as a double integral of sqrt(x^2 + y^2) dx dy over the area of the pixel
        self.distance_layer[self.center, self.center] = 0.3826*self.pixel_size 

        # determine the bearing of each pixel from the centre of the image
        self.bearing_layer = torch.from_numpy(np.arctan2(self.center - y, x - self.center)).float()
        self.device = params.device


    # Gamma densities (on the log-scale) for the mixture distribution
    def gamma_density(self, x, shape, scale):
        # Ensure all tensors are on the same device as x
        shape = shape.to(x.device)
        scale = scale.to(x.device)
        # return -1*torch.lgamma(shape) -shape*torch.log(scale) + (shape - 1)*torch.log(x) - x/scale
        
        # to account for change of variables
        return (-1*torch.lgamma(shape) -shape*torch.log(scale) + (shape - 1)*torch.log(x) - x/scale) - torch.log(x)

    # log von Mises densities (on the log-scale) for the mixture distribution
    def vonmises_density(self, x, kappa, vm_mu):
        # Ensure all tensors are on the same device as x
        kappa = kappa.to(x.device)
        vm_mu = vm_mu.to(x.device)
        return kappa*torch.cos(x - vm_mu) - 1*(np.log(2*torch.pi) + torch.log(torch.special.i0(kappa)))


    def forward(self, x, bearing):

        # parameters of the first mixture distribution
        # x are the outputs from the fully connected layers (vector of movement parameters)
        # we therefore need to extract the appropriate parameters 
        # the locations are not specific to any specific parameters, as long as any aren't extracted more than once 

        # Gamma distributions

        # pull out the parameters of the first gamma distribution and exponentiate them to ensure they are positive
        # the unsqueeze function adds a new dimension to the tensor
        # we do this twice to match the dimensions of the distance_layer, 
        # and then repeat the parameter value across a grid, such that the density can be calculated at every cell/pixel
        gamma_shape1 = torch.exp(x[:, 0]).unsqueeze(0).unsqueeze(0)
        gamma_shape1 = gamma_shape1.repeat(self.image_dim, self.image_dim, 1)
        # this just changes the order of the dimensions to match the distance_layer
        gamma_shape1 = gamma_shape1.permute(2, 0, 1)

        gamma_scale1 = torch.exp(x[:, 1]).unsqueeze(0).unsqueeze(0)
        gamma_scale1 = gamma_scale1.repeat(self.image_dim, self.image_dim, 1)
        gamma_scale1 = gamma_scale1.permute(2, 0, 1)

        gamma_weight1 = torch.exp(x[:, 2]).unsqueeze(0).unsqueeze(0)
        gamma_weight1 = gamma_weight1.repeat(self.image_dim, self.image_dim, 1)
        gamma_weight1 = gamma_weight1.permute(2, 0, 1)

        # parameters of the second mixture distribution
        gamma_shape2 = torch.exp(x[:, 3]).unsqueeze(0).unsqueeze(0)
        gamma_shape2 = gamma_shape2.repeat(self.image_dim, self.image_dim, 1)
        gamma_shape2 = gamma_shape2.permute(2, 0, 1)

        gamma_scale2 = torch.exp(x[:, 4]).unsqueeze(0).unsqueeze(0)
        gamma_scale2 = gamma_scale2 * 500 ### transform the scale parameter so it can be estimated near the same range as the other parameters
        gamma_scale2 = gamma_scale2.repeat(self.image_dim, self.image_dim, 1)
        gamma_scale2 = gamma_scale2.permute(2, 0, 1)

        gamma_weight2 = torch.exp(x[:, 5]).unsqueeze(0).unsqueeze(0)
        gamma_weight2 = gamma_weight2.repeat(self.image_dim, self.image_dim, 1)
        gamma_weight2 = gamma_weight2.permute(2, 0, 1)

        # Apply softmax to the mixture weights to ensure they sum to 1
        gamma_weights = torch.stack([gamma_weight1, gamma_weight2], dim=0)
        gamma_weights = torch.nn.functional.softmax(gamma_weights, dim=0)
        gamma_weight1 = gamma_weights[0]
        gamma_weight2 = gamma_weights[1]

        # calculation of Gamma densities
        gamma_density_layer1 = self.gamma_density(self.distance_layer, gamma_shape1, gamma_scale1).to(self.device)
        gamma_density_layer2 = self.gamma_density(self.distance_layer, gamma_shape2, gamma_scale2).to(self.device)

        # combining both densities to create a mixture distribution using logsumexp
        logsumexp_gamma_corr = torch.max(gamma_density_layer1, gamma_density_layer2)
        gamma_density_layer = logsumexp_gamma_corr + torch.log(gamma_weight1 * torch.exp(gamma_density_layer1 - logsumexp_gamma_corr) + 
                                                               gamma_weight2 * torch.exp(gamma_density_layer2 - logsumexp_gamma_corr))
        # print(torch.sum(gamma_density_layer))
        # print(torch.sum(torch.exp(gamma_density_layer)))


        ## Von Mises Distributions

        # calculate the new bearing from the turning angle
        # takes in the bearing from the previous step and adds the turning angle, which is estimated by the model
        # we do not exponentiate the von Mises mu parameters as we want to allow them to be negative
        bearing_new1 = x[:, 6] + bearing[:, 0]

        # the new bearing becomes the mean of the von Mises distribution
        vonmises_mu1 = bearing_new1.unsqueeze(0).unsqueeze(0)
        vonmises_mu1 = vonmises_mu1.repeat(self.image_dim, self.image_dim, 1)
        vonmises_mu1 = vonmises_mu1.permute(2, 0, 1)

        # parameters of the first von Mises distribution
        vonmises_kappa1 = torch.exp(x[:, 7]).unsqueeze(0).unsqueeze(0)
        vonmises_kappa1 = vonmises_kappa1.repeat(self.image_dim, self.image_dim, 1)
        vonmises_kappa1 = vonmises_kappa1.permute(2, 0, 1)

        vonmises_weight1 = torch.exp(x[:, 8]).unsqueeze(0).unsqueeze(0)
        vonmises_weight1 = vonmises_weight1.repeat(self.image_dim, self.image_dim, 1)
        vonmises_weight1 = vonmises_weight1.permute(2, 0, 1)

        # vm_mu and weight for the second von Mises distribution
        bearing_new2 = x[:, 9] + bearing[:, 0]

        vonmises_mu2 = bearing_new2.unsqueeze(0).unsqueeze(0)
        vonmises_mu2 = vonmises_mu2.repeat(self.image_dim, self.image_dim, 1)
        vonmises_mu2 = vonmises_mu2.permute(2, 0, 1)

        # parameters of the second von Mises distribution
        vonmises_kappa2 = torch.exp(x[:, 10]).unsqueeze(0).unsqueeze(0)
        vonmises_kappa2 = vonmises_kappa2.repeat(self.image_dim, self.image_dim, 1)
        vonmises_kappa2 = vonmises_kappa2.permute(2, 0, 1)

        vonmises_weight2 = torch.exp(x[:, 11]).unsqueeze(0).unsqueeze(0)
        vonmises_weight2 = vonmises_weight2.repeat(self.image_dim, self.image_dim, 1)
        vonmises_weight2 = vonmises_weight2.permute(2, 0, 1)

        # Apply softmax to the weights
        vonmises_weights = torch.stack([vonmises_weight1, vonmises_weight2], dim=0)
        vonmises_weights = torch.nn.functional.softmax(vonmises_weights, dim=0)
        vonmises_weight1 = vonmises_weights[0]
        vonmises_weight2 = vonmises_weights[1]

        # calculation of von Mises densities
        vonmises_density_layer1 = self.vonmises_density(self.bearing_layer, vonmises_kappa1, vonmises_mu1).to(self.device)
        vonmises_density_layer2 = self.vonmises_density(self.bearing_layer, vonmises_kappa2, vonmises_mu2).to(self.device)

        # combining both densities to create a mixture distribution using the logsumexp trick
        logsumexp_vm_corr = torch.max(vonmises_density_layer1, vonmises_density_layer2)
        vonmises_density_layer = logsumexp_vm_corr + torch.log(vonmises_weight1 * torch.exp(vonmises_density_layer1 - logsumexp_vm_corr) + vonmises_weight2 * torch.exp(vonmises_density_layer2 - logsumexp_vm_corr))
        # print(torch.sum(vonmises_density_layer))
        # print(torch.sum(torch.exp(vonmises_density_layer)))

        # combining the two distributions
        movement_grid = gamma_density_layer + vonmises_density_layer # Gamma and von Mises densities are on the log-scale

        # normalise (on the log-scale using the log-sum-exp trick) before combining with the habitat predictions
        movement_grid = movement_grid - torch.logsumexp(movement_grid, dim = (1, 2), keepdim = True)
        # print('Movement grid norm ', torch.sum(movement_grid))
        # print(torch.sum(torch.exp(movement_grid)))

        return movement_grid



"""
## Scalar to grid block

This block takes any scalar value (e.g., time of day, day of year) and converts 
it to a 2D image, with the same values for all pixels. 

This is so that the scalar values can be used as input to the convolutional layers.

"""
class Scalar_to_Grid_Block(nn.Module):
    def __init__(self, params):
        super(Scalar_to_Grid_Block, self).__init__()

        # define the parameters
        self.batch_size = params.batch_size
        self.image_dim = params.image_dim
        self.device = params.device

    def forward(self, x):

        # how many scalar values are being passed in
        num_scalars = x.shape[1]
        # expand the scalar values to the spatial dimensions of the image
        scalar_map = x.view(x.shape[0], num_scalars, 1, 1).expand(x.shape[0], num_scalars, self.image_dim, self.image_dim)

        # return the scalar maps
        return scalar_map


"""
## Combine the blocks into the deepSSF model

Here is where we combine the blocks into a model. Similarly to the previous blocks, 
the model is a Python class that inherits from `torch.nn.Module`, 
which combines other `torch.nn.Module` modules. 

For example, we can instantiate the habitat selection convolution block 
using `self.conv_habitat = Conv2d_block_spatial(params)` in the `__init__` 
method (the 'constructor' for a class). We can now access that block using 
`self.conv_habitat` in the forward method. 

In the forward method, we pass the input data through the habitat selection 
convolution block using `output_habitat = self.conv_habitat(all_spatial)`, 
where `all_spatial` is the input data, which is a combination of the spatial 
covariates and the scalar values converted to 2D images.

First we instantiate the blocks, and then define the forward method, which 
defines the data flow through the network during inference or training.

"""
class ConvJointModel(nn.Module):
    def __init__(self, params):
        """
        ConvJointModel:
        - Initializes blocks for scalar-to-grid transformation, 
          habitat convolution, movement convolution + movement fully connected, and final parameter-to-grid transformation.
        - Accepts parameters from the params object, which we will define later.
        """
        super(ConvJointModel, self).__init__()

        # Block to convert scalar features into grid-like (spatial) features
        self.scalar_grid_output = Scalar_to_Grid_Block(params)

        # Convolutional block for habitat selection
        self.conv_habitat = Conv2d_block_spatial(params)

        # Convolutional block for movement extraction (output fed into fully connected layers)
        self.conv_movement = Conv2d_block_toFC(params)

        # Fully connected block for movement
        self.fcn_movement_all = FCN_block_all_movement(params)

        # Converts movement distribution parameters into a grid (the 2D movement kernel)
        self.movement_grid_output = Params_to_Grid_Block_ChV(params)

        # Device information from params (e.g., CPU or GPU)
        self.device = params.device

    def forward(self, x):
        """
        Forward pass:
        1. Extract scalar data and convert to grid features.
        2. Concatenate the newly created scalar-based grids with spatial data.
        3. Pass this combined input through separate sub-networks for habitat and movement.
        4. Convert movement parameters to a grid, then stack the habitat and movement outputs.
        """
        # x contains:
        # - spatial_data_x (image-like layers)
        # - scalars_to_grid (scalar features needing conversion)
        # - bearing_x (the bearing from the previous time step, the turning angle is estimated as the deviation from this)
        spatial_data_x = x[0]
        scalars_to_grid = x[1]
        bearing_x = x[2]

        # Convert scalar data to spatial (grid) form
        scalar_grids = self.scalar_grid_output(scalars_to_grid)

        # Combine the original spatial data with the newly generated scalar grids
        all_spatial = torch.cat([spatial_data_x, scalar_grids], dim=1)

        # HABITAT SUBNETWORK
        # Convolutional feature extraction for habitat selection
        output_habitat = self.conv_habitat(all_spatial)

        # MOVEMENT SUBNETWORK
        # Convolutional feature extraction (different architecture for movement)
        conv_movement = self.conv_movement(all_spatial)

        # Fully connected layers for movement (processing both spatial features and any extras)
        output_movement = self.fcn_movement_all(conv_movement)

        # Transform the movement parameters into a grid, using bearing information
        output_movement = self.movement_grid_output(output_movement, bearing_x)

        # Combine (stack) habitat and movement outputs without merging them
        output = torch.stack((output_habitat, output_movement), dim=-1)

        return output


"""
## Set the parameters for the model which will be specified in a dictionary

This Python class serves as a simple parameter container for a model that involves both 
spatial (e.g., convolutional layers) and non-spatial inputs. 
It captures all relevant hyperparameters and settings—such as image dimensions, 
kernel sizes, and fully connected layer dimensions—along with the target device (CPU or GPU). 
This structure allows easy configuration of the model without scattering parameters throughout the code.

"""
class ModelParams():
    def __init__(self, dict_params):
        self.batch_size = dict_params["batch_size"]
        self.image_dim = dict_params["image_dim"]
        self.pixel_size = dict_params["pixel_size"]
        self.batch_size = dict_params["batch_size"]
        self.dim_in_nonspatial_to_grid = dict_params["dim_in_nonspatial_to_grid"]
        self.dense_dim_in_nonspatial = dict_params["dense_dim_in_nonspatial"]
        self.dense_dim_hidden = dict_params["dense_dim_hidden"]
        self.batch_size = dict_params["batch_size"]
        self.dense_dim_in_all = dict_params["dense_dim_in_all"]
        self.batch_size = dict_params["batch_size"]
        self.input_channels = dict_params["input_channels"]
        self.output_channels = dict_params["output_channels"]
        self.kernel_size = dict_params["kernel_size"]
        self.stride = dict_params["stride"]
        self.kernel_size_mp = dict_params["kernel_size_mp"]
        self.stride_mp = dict_params["stride_mp"]
        self.padding = dict_params["padding"]
        self.image_dim = dict_params["image_dim"]
        self.num_movement_params = dict_params["num_movement_params"]
        self.dropout = dict_params["dropout"]
        self.device = dict_params["device"]


### These are defined in each of the scripts that use the model

# """
# ## Define the parameters for the model

# Here we enter the specific parameter values and hyperparameters for the model. 
# These are the values that will be used to instantiate the model.

# """
# # Define the parameters for the model


