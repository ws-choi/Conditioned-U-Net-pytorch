import torch
import torch.nn as nn
import pytorch_lightning as pl


class u_net_conv_block(pl.LightningModule):
    def __init__(self, bn_layer, conv_layer, activation):
        super(u_net_conv_block, self).__init__()
        self.bn_layer = bn_layer
        self.conv_layer = conv_layer
        self.activation = activation
        self.in_channels = self.conv_layer.conv.in_channels
        self.out_channels = self.conv_layer.conv.out_channels

    def forward(self, x):
        x = self.bn_layer(self.conv_layer(x))
        return self.activation(x)


class u_net_deconv_block(pl.LightningModule):

    def __init__(self, deconv_layer, bn_layer, activation, dropout):
        super(u_net_deconv_block, self).__init__()
        self.bn_layer = bn_layer
        self.deconv_layer = deconv_layer
        self.dropout = nn.Dropout() if dropout else nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.bn_layer(self.deconv_layer(x))
        x = self.dropout(x)
        return self.activation(x)


class Conv2d_same(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2)):
        super(Conv2d_same, self).__init__()
        padding = [((k - s + 1) // 2) for k, s in zip(kernel_size, stride)]
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2d_same(pl.LightningModule):

    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2)):
        super(ConvTranspose2d_same, self).__init__()

        # Assuming dilation = 1,
        # H_out = (H_in-1) * stride - 2 * padding + (kernel_size -1) + output_padding + 1
        # We want to make H_out = H_in * stride => Thus,
        # H_in * stride = (H_in-1) * stride - 2 * padding + (kernel_size -1) + output_padding + 1
        # 0 = (0-1) * stride - 2 * padding + kernel_size  + output_padding
        # 2 * padding = -stride + (kernel_size + output_padding)
        # padding = (- stride + kernel_size + output_padding   )/2

        output_padding = [abs(k % 2 - s % 2) for k, s in zip(kernel_size, stride)]
        padding = [(k - s + o) // 2 for k, s, o in zip(kernel_size, stride, output_padding)]
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding
        )

    def forward(self, x):
        return self.deconv(x)


class UNET(pl.LightningModule):

    def __init__(self,
                 n_layers,
                 input_channels,
                 filters_layer_1,
                 kernel_size=(5, 5),
                 stride=(2, 2),
                 encoder_activation=nn.LeakyReLU,
                 decoder_activation=nn.ReLU,
                 last_activation=nn.Sigmoid,
                 ):

        super(UNET, self).__init__()
        self.n_layers = n_layers
        self.input_channels = input_channels
        self.filters_layer_1 = filters_layer_1
        encoder_layers = []

        # Encoder
        encoders = []
        for i in range(n_layers):
            output_channels = filters_layer_1 * (2 ** i)
            encoders.append(
                u_net_conv_block(
                    conv_layer=Conv2d_same(input_channels, output_channels, kernel_size, stride),
                    bn_layer=nn.BatchNorm2d(output_channels),
                    activation=encoder_activation()
                )
            )
            input_channels = output_channels
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []
        for i in range(n_layers):
            # parameters each decoder layer
            is_final_block = i == n_layers - 1  # the las layer is different
            # not dropout in the first block and the last two encoder blocks
            dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
            # for getting the number of filters
            encoder_layer = self.encoders[n_layers - i - 1]
            skip = i > 0  # not skip in the first encoder block

            input_channels = encoder_layer.out_channels
            if skip:
                input_channels *= 2

            if is_final_block:
                output_channels = self.input_channels
                activation = last_activation
            else:
                output_channels = encoder_layer.in_channels
                activation = decoder_activation

            decoders.append(
                u_net_deconv_block(
                    deconv_layer=ConvTranspose2d_same(input_channels, output_channels, kernel_size, stride),
                    bn_layer=nn.BatchNorm2d(output_channels),
                    activation=activation(),
                    dropout=dropout
                )
            )

            self.decoders = nn.ModuleList(decoders)

    def forward(self, input_spec):

        x = input_spec
        # Encoding Phase
        encoder_outputs = []
        for encoder in self.encoders:
            encoder_outputs.append(encoder(x))
            x = encoder_outputs[-1]

        # Decoding Phase
        x = self.decoders[0](x)
        for decoder, x_encoded in zip(self.decoders[1:], reversed(encoder_outputs[:-1])):
            x = decoder(torch.cat([x, x_encoded], dim=-3))

        return x
