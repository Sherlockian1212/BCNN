import torch
import torch.nn.functional as F
from torch.nn import Parameter

from layers.misc.moduleWrapper import moduleWrapper
from metrics.KL import KL

class BCNN(moduleWrapper):
    def __init__(self, inp_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 priors=None):
        super(BCNN, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.W_mu = Parameter(torch.Tensor(out_channels, inp_channels, *self.kernel_size)).to(self.device)
        self.W_rho = Parameter(torch.Tensor(out_channels, inp_channels, *self.kernel_size)).to(self.device)
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels)).to(self.device)
            self.bias_rho = Parameter(torch.Tensor(out_channels)).to(self.device)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.resetParameter()

    def resetParameter(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.W_mu.data.normal_(*self.posterior_mu_initial)
            self.W_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_esp = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_esp * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input=x,
                        weight=weight,
                        bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def klLoss(self):
        kl = KL(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        return kl