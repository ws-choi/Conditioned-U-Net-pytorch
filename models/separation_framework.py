from abc import ABCMeta
from argparse import ArgumentParser

import numpy as np
import pydub
import pytorch_lightning as pl
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as f
import wandb
from pytorch_lightning.loggers import WandbLogger
import models.cunet_model as cunet
from models import fourier


def get_estimation(idx, target_name, estimation_dict):
    estimated = estimation_dict[target_name][idx]
    if len(estimated) == 0:
        raise NotImplementedError
    estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
    return estimated


class Conditional_Source_Separation(pl.LightningModule, metaclass=ABCMeta):

    def __init__(self, n_fft, hop_length, num_frame, optimizer, lr, dev_mode, **kwargs):
        super(Conditional_Source_Separation, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        self.lr = lr
        self.optimizer = optimizer

        self.target_names = ['vocals', 'drums', 'bass', 'other']
        self.valid_estimation_dict = {}

        self.dev_mode = dev_mode

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
        loss = f.mse_loss(target, target_hat)
        result = pl.TrainResult(loss)
        result.log('loss/train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                   reduce_fx=torch.mean)
        return result

    # Validation Process
    def on_validation_epoch_start(self):
        for target_name in self.target_names:
            self.valid_estimation_dict[target_name] = {mixture_idx: {}
                                                       for mixture_idx
                                                       in range(14)}

    def validation_step(self, batch, batch_idx):
        mixtures, mixture_ids, window_offsets, input_conditions, target_names, targets = batch

        estimated_targets = self.separate(mixtures, input_conditions)[:, self.hop_length:-self.hop_length]
        targets = targets[:, self.hop_length:-self.hop_length]

        for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):
            self.valid_estimation_dict[target_name][mixture_idx.item()][
                window_offset.item()] = estimated_target.detach().cpu().numpy()

        # SDR - like Loss
        s_targets = ((targets * estimated_targets).sum(axis=-2, keepdims=True) / (
                (targets ** 2).sum(axis=-2, keepdims=True) + 1e-8)) * targets
        distortion = estimated_targets - s_targets

        loss = (((s_targets ** 2).sum(-2) + 1e-8).log() - ((distortion ** 2).sum(-2) + 1e-8).log()).mean()

        # large value of SDR means good performance, so that we take the negative of sdr for the validation loss
        loss = -1 * loss

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('loss/val_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                   reduce_fx=torch.mean)
        return result

    def on_validation_epoch_end(self):
        val_ids = [0] if self.dev_mode else [0, 1, 2]
        for idx in val_ids:
            estimation = {}
            for target_name in self.target_names:
                estimation[target_name] = get_estimation(idx, target_name, self.valid_estimation_dict)
                if estimation[target_name] is not None:
                    estimation[target_name] = estimation[target_name].astype(np.float32)

                    if self.current_epoch > 10 and isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, target_name): [
                            wandb.Audio(estimation[target_name][44100 * 60:44100 * 65],
                                        caption='{}_{}'.format(idx, target_name),
                                        sample_rate=44100)]})

    def on_test_epoch_start(self):

        import os
        output_folder = 'museval_output'
        if os.path.exists(output_folder):
            os.rmdir(output_folder)
        os.mkdir(output_folder)

        self.valid_estimation_dict = None
        self.test_estimation_dict = {}
        self.test_true_dict = {}

        self.musdb_test = self.test_dataloader().dataset
        num_tracks = self.musdb_test.num_tracks
        for target_name in self.target_names:
            self.test_estimation_dict[target_name] = {mixture_idx: {}
                                                      for mixture_idx
                                                      in range(num_tracks)}

            self.test_true_dict[target_name] = {mixture_idx: self.musdb_test.get_audio(mixture_idx, target_name)
                                                for mixture_idx in range(num_tracks)}

    def test_step(self, batch, batch_idx):
        mixtures, mixture_ids, window_offsets, input_conditions, target_names = batch

        estimated_targets = self.separate(mixtures, input_conditions)[:, self.hop_length:-self.hop_length]

        for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):
            self.test_estimation_dict[target_name][mixture_idx.item()][
                window_offset.item()] = estimated_target.detach().cpu().numpy()

            # print(mixture_idx.item(), ':', target_name, window_offset.item() )

        # pl.metrics.converters.sync_ddp()
        return torch.zeros(0)

    def on_test_epoch_end(self):

        import museval
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')
        idx_list = [3, 2, 1, 0] if self.dev_mode else range(self.musdb_test.num_tracks)

        for idx in idx_list:
            estimation = {}
            for target_name in self.target_names:
                estimation[target_name] = get_estimation(idx, target_name, self.test_estimation_dict)
                if estimation[target_name] is not None:
                    estimation[target_name] = estimation[target_name].astype(np.float32)

            # Real SDR
            if len(estimation) == len(self.target_names):
                true_targets = [self.test_true_dict[target_name][idx] for target_name in self.target_names]
                track_length = true_targets[0].shape[0]
                estimated_targets = [estimation[target_name][:track_length] for target_name in self.target_names]

                if track_length > estimated_targets[0].shape[0]:
                    raise NotImplementedError
                else:
                    estimated_targets_dict = {target_name: estimation[target_name][:track_length] for target_name in
                                              self.target_names}
                    track_score = museval.eval_mus_track(
                        self.musdb_test.musdb_test[idx],
                        estimated_targets_dict
                    )

                    score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                        ['target', 'metric'])['score']\
                        .median().to_dict()

                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log(
                            {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})

                    else:
                        print(track_score)

                    results.add_track(track_score)

        if isinstance(self.logger, WandbLogger):

            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            self.logger.experiment.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )
        else:
            print(results)

    def export_mp3(self, idx, target_name):
        estimated = self.test_estimation_dict[target_name][idx]
        estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
        soundfile.write('tmp_output.wav', estimated, samplerate=44100)
        audio = pydub.AudioSegment.from_wav('tmp_output.wav')
        audio.export('{}_estimated/output_{}.mp3'.format(idx, target_name))

    def separate(self, input_signal, input_condition) -> torch.Tensor:
        pass

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')

        return parser


class CUNET_Framework(Conditional_Source_Separation):

    @staticmethod
    def get_arg_keys():
        return ['n_fft', 'hop_length', 'num_frame', 'spec_type', 'spec_est_mode', 'optimizer', 'lr', 'dev_mode'] \
               + cunet.CUNET.get_arg_keys()

    def __init__(self, n_fft, hop_length, num_frame, spec_type, spec_est_mode, **kwargs):
        super(CUNET_Framework, self).__init__(n_fft, hop_length, num_frame, **kwargs)

        self.save_hyperparameters(*CUNET_Framework.get_arg_keys())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        self.magnitude_based = spec_type == "magnitude"
        self.masking_based = spec_est_mode == "masking"
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)

        cunet_args = cunet.CUNET.get_arg_keys()
        self.spec2spec = cunet.CUNET(**{key: kwargs[key] for key in cunet_args})

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
        if self.magnitude_based:
            return self.stft.to_mag(input_signal).transpose(-1, -3)[..., 1:]
        else:
            raise NotImplementedError

    def separate(self, input_signal, input_condition) -> torch.Tensor:

        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)
            output_spec = self.spec2spec(input_spec[..., 1:], input_condition)

            if self.masking_based:
                output_spec = input_spec[..., 1:] * output_spec
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        output_spec = torch.cat([input_spec[..., :1], output_spec], dim=-1)
        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            raise NotImplementedError

        return restored

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

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

        return Conditional_Source_Separation.add_model_specific_args(parser)
