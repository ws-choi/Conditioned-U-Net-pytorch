import os
import time
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data.dataloaders import DataProvider
from models.separation_framework import CUNET_Framework
from models.utils import get_model


def main(args):
    dict_args = vars(args)

    model_name = dict_args['model_name']

    model = get_model(model_name, dict_args)

    checkpoint_path = dict_args['checkpoints_path']
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    if dict_args['log_system'] == 'wandb':

        logger = WandbLogger(project='source_separation', tags=model_name, offline=False, id=dict_args['run_id'])
        logger.log_hyperparams(model.hparams)
        logger.watch(model, log='all')

    elif dict_args['log_system'] == 'tensorboard':
        raise NotImplementedError
    else:
        logger = True  # default

    model_dir = checkpoint_path +dict_args['model_name']
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    ckpt_dir = '{}/{}'.format(model_dir, dict_args['run_id'])
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_dir,
        save_top_k=dict_args['save_top_k'],
        verbose=False,
        monitor='val_loss',
        prefix=dict_args['model_name'] + '_',
        save_last=True,
        save_weights_only= True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=dict_args['patience'],
        verbose=False

    )
    if dict_args['float16']:
        trainer = Trainer(
            gpus=dict_args['gpus'],
            precision=16,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            distributed_backend=dict_args['distributed_backend']
        )
    else:
        trainer = Trainer(
            gpus=dict_args['gpus'],
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            distributed_backend=dict_args['distributed_backend']
        )

    data_provider = DataProvider(**dict_args)

    n_fft, hop_length, num_frame = [dict_args[key] for key in ['n_fft', 'hop_length', 'num_frame']]

    train_dataloader = data_provider.get_train_dataloader(n_fft, hop_length, num_frame)
    valid_dataloader = data_provider.get_valid_dataloader(n_fft, hop_length, num_frame)
    # test_dataloader = data_provider.get_test_dataloader(n_fft, hop_length, num_frame)

    trainer.fit(model, train_dataloader, valid_dataloader)
    # trainer.test(model, test_dataloader)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_name', type=str, default='cunet')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints/')
    parser.add_argument('--log_system', type=str, default=True)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--run_id', type=str, default=str(time.time()))

    parser.add_argument('--save_top_k', type=int, default=3)
    parser.add_argument('--patience', type=int, default=25)

    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "cunet":
        parser = CUNET_Framework.add_model_specific_args(parser)

    parser = DataProvider.add_data_provider_args(parser)
    args = parser.parse_args()

    # train
    main(args)
