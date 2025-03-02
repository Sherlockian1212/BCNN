from layers.misc.moduleWrapper import moduleWrapper

class flattenLayer(moduleWrapper):
    def __init__(self, num_features):
        super(flattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)