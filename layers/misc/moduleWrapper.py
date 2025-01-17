from torch import nn

class moduleWrapper(nn.Module):
    def __init__(self):
        super(moduleWrapper, self).__init__()

    def setFlag(self, flag_name, value):
        setattr(self, flag_name, value)
        for child in self.children():
            if hasattr(child, 'setFlag'):
                child.setFlag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'klLoss'):
                kl = kl + module.klLoss()

        return x,kl