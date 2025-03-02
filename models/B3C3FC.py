import torch.nn as nn
from layers.BCNN import BCNN
from layers.BLinear import BLinear
from layers.misc.flattenLayer import flattenLayer
from layers.misc.moduleWrapper import moduleWrapper

"""
3 Convolution and 3 FC layers with Bayesian layers.
"""
class B3C3FC(moduleWrapper):
    def __init__(self, inputs, outputs, priors, activation_type="softplus"):
        super(B3C3FC, self).__init__()
        self.activation = activation_type
        self.num_classes = outputs
        self.priors = priors

        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        elif activation_type == "softplus":
            self.activation = nn.Softplus()  # Thêm đối tượng Softplus()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        self.act1 = self.activation
        self.conv1 = BCNN(inputs, 32, 5, padding=2, bias=True, priors=None)
        self.act1 = self.activation
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BCNN(32, 64, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = self.activation
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BCNN(64, 128, 5, padding=1, bias=True, priors=self.priors)
        self.act3 = self.activation
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = flattenLayer(num_features=2 * 2 * 128)
        self.fc1 = BLinear(2 * 2 * 128, 1000, bias=True, priors=self.priors)
        self.act4 = self.activation

        self.fc2 = BLinear(1000, 1000, bias=True, priors=self.priors)
        self.act5 = self.activation

        self.fc3 = BLinear(1000, outputs, bias=True, priors=self.priors)
