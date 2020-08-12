import pytorch_lightning as pl
import torch.nn as nn


class dense_control_block(pl.LightningModule):

    def __init__(self, input_dim, num_layer, activation=nn.ReLU):
        super(dense_control_block, self).__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.activation = activation

        linear_list = []
        dims = [input_dim * (4 ** i) for i in range(num_layer)]
        for i, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
            extra = i != 0
            linear_list.append(nn.Linear(in_features, out_features))
            linear_list.append(activation())

            if extra:
                linear_list.append(nn.Dropout())
                linear_list.append(nn.BatchNorm1d(out_features))

        self.linear = nn.Sequential(*linear_list)
        self.last_dim = dims[-1]

    def forward(self, x_condition):
        return self.linear(x_condition)


class dense_control_model(pl.LightningModule):
    def __init__(self, dense_control_block, output_features, split, gamma_activation=nn.Identity, beta_activation=nn.Identity):
        super(dense_control_model, self).__init__()
        self.dense_control_block = dense_control_block
        self.split = split
        self.linear_gamma = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, output_features),
            gamma_activation()
        )
        self.linear_beta = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, output_features),
            beta_activation()
        )

    def forward(self, x):
        x = self.dense_control_block(x)
        return self.split(self.linear_gamma(x)), self.split(self.linear_beta(x))


