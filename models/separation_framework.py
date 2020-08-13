from abc import ABCMeta
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.cunet_model as cunet
from data.data_loader import MusdbLoader, MusdbTrainSet, MusdbValidSet
from models import fourier


class Conditional_Source_Separation(pl.LightningModule, metaclass=ABCMeta):

    def __init__(self, n_fft, hop_length, num_frame, musdb_root, no_data_cache, batch_size, dev_mode, optimizer, lr, num_workers, pin_memory, **kwargs):
        super(Conditional_Source_Separation, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        self.musdb_root = musdb_root
        self.data_cache = not no_data_cache
        self.batch_size = batch_size
        self.dev_mode = dev_mode

        self.lr = lr
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.musdb_loader = MusdbLoader(musdb_root=self.musdb_root)
        if self.data_cache:
            self.cached_musdb_train = MusdbTrainSet(self.musdb_loader.musdb_train,
                                                    n_fft=self.n_fft,
                                                    hop_length=self.hop_length,
                                                    num_frame=self.num_frame,
                                                    cache_mode=True,
                                                    dev_mode=self.dev_mode)
            self.cached_musdb_valid = MusdbValidSet(self.musdb_loader.musdb_valid,
                                                    n_fft=self.n_fft,
                                                    hop_length=self.hop_length,
                                                    num_frame=self.num_frame,
                                                    cache_mode=True,
                                                    dev_mode=self.dev_mode)

    def train_dataloader(self) -> DataLoader:
        if self.data_cache:
            musdb_train = self.cached_musdb_train
        else:
            musdb_train = MusdbTrainSet(self.musdb_loader.musdb_train,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop_length,
                                        num_frame=self.num_frame,
                                        cache_mode=False)

        return DataLoader(musdb_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        if self.data_cache:
            musdb_valid = self.cached_musdb_valid
        else:
            musdb_valid = MusdbValidSet(self.musdb_loader.musdb_valid,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop_length,
                                        num_frame=self.num_frame,
                                        cache_mode=False)

        return DataLoader(musdb_valid,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def configure_optimizers(self):

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop
        else:
            optimizer = torch.optim.Adam

        return optimizer(self.parameters(), lr=float(self.lr))

    def training_step(self, batch, batch_idx):
        mixture_signal, target_signal, condition = batch
        target = self.to_spec(target_signal)
        target_hat = self.forward(mixture_signal, condition)
        loss = F.mse_loss(target, target_hat)
        result = pl.TrainResult(loss)
        result.log('loss/train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                   reduce_fx=torch.mean)
        return result

    def validation_step(self, batch, batch_idx):
        mixture_signal, target_signal, condition = batch
        target = self.to_spec(target_signal)
        target_hat = self.forward(mixture_signal, condition)
        loss = F.mse_loss(target, target_hat)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('loss/val_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                   reduce_fx=torch.mean)
        return result

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--musdb_root', type=str, default='data/musdb18_wav/')
        parser.add_argument('--musdb_is_wav', type=bool, default=False)
        parser.add_argument('--no_data_cache', type=bool, default=False)
        parser.add_argument('--dev_mode', type=bool, default=False)

        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--pin_memory', type=bool, default=False)

        return parser


class CUNET_Framework(Conditional_Source_Separation):

    @staticmethod
    def get_arg_keys():
        return ['n_fft', 'hop_length', 'num_frame', 'spec_type', 'spec_est_mode'] + cunet.CUNET.get_arg_keys()

    def __init__(self, n_fft, hop_length, num_frame, spec_type, spec_est_mode, **kwargs):
        super(CUNET_Framework, self).__init__(n_fft, hop_length, num_frame, **kwargs)

        self.save_hyperparameters(*CUNET_Framework.get_arg_keys())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        self.magnitude_based = spec_type == "magnitude"
        self.masking_based = spec_est_mode == "masking"
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)

        CUNET_Args = cunet.CUNET.get_arg_keys()
        self.spec2spec = cunet.CUNET(**{key: kwargs[key] for key in CUNET_Args})



        self.init_weights()

    def forward(self, input_signal, input_condition):
        input_spec = self.to_spec(input_signal)
        output_spec = self.spec2spec(input_spec, input_condition)

        if self.masking_based:
            output_spec = input_spec * output_spec

        return output_spec

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)

    def to_spec(self, input_signal) -> torch.Tensor:
        return self.stft.to_mag(input_signal).transpose(-1, -3)[..., 1:]

    def separate(self, input_signal, input_condition) -> torch.Tensor:
        mag, phase = self.stft.to_mag_phase(input_signal)
        mag_hat = self.forward(mag.transpose(-1, -3), input_condition)
        return self.stft.restore_mag_phase(mag_hat.transpose(-1, -3), phase)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = Conditional_Source_Separation.add_model_specific_args(parser)

        parser.add_argument('--n_fft', type=int, default=1024)
        parser.add_argument('--hop_length', type=int, default=256)
        parser.add_argument('--num_frame', type=int, default=128)
        parser.add_argument('--spec_type', type=str, default='magnitude')
        parser.add_argument('--spec_est_mode', type=str, default='masking')

        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--input_channels', type=int, default=2)
        parser.add_argument('--filters_layer_1', type=int, default=16)
        parser.add_argument('--kernel_size', default=(5, 5))
        parser.add_argument('--stride', default=(2, 2))
        parser.add_argument('--film_type', type=str, default='simple')
        parser.add_argument('--control_type', type=str, default='dense')
        parser.add_argument('--encoder_activation', type=str, default='leaky_relu')
        parser.add_argument('--decoder_activation', type=str, default='relu')
        parser.add_argument('--last_activation', type=str, default='sigmoid')
        parser.add_argument('--control_input_dim', type=int, default=4)
        parser.add_argument('--control_n_layer', type=int, default=4)

        return parser
