from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer

from data.dataloaders import DataProvider
from models.separation_framework import CUNET_Framework
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os


def main(args):
    dict_args = vars(args)
    model_name = dict_args['model_name']

    if model_name == 'cunet':
        model = CUNET_Framework(**dict_args)
    else:
        raise NotImplementedError

    checkpoint_path = dict_args['checkpoints_path']
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    if dict_args['run_id'] is None:
        import time
        dict_args['run_id'] = model_name + "_" + str(time.time())
    if dict_args['log_system'] == 'wandb':
        logger = WandbLogger(project='source_separation', tags=model_name, offline=False, id=dict_args['run_id'])
        logger.log_hyperparams(model.hparams)
        logger.watch(model, log='all')
        checkpoint_path += '/' + logger.version

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

    elif dict_args['log_system'] == 'tensorboard':
        if not os.path.exists(dict_args['tensorboard_path']):
            os.mkdir(dict_args['tensorboard_path'])
        logger = pl_loggers.TensorBoardLogger(dict_args['tensorboard_path'], name=model_name)
    else:
        logger = True  # default

    distributed_backend = 'ddp' if dict_args['gpus'] > 1 else None

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_top_k=10,
        verbose=False,
        monitor='val_loss',
        prefix=dict_args['model_name'] + '_'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=25,
        verbose=False

    )
    if dict_args['float16']:
        trainer = Trainer(
            gpus=temp_args.gpus,
            precision=16,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            distributed_backend=distributed_backend
        )
    else:
        trainer = Trainer(
            gpus=temp_args.gpus,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            distributed_backend=distributed_backend
        )

    data_provider = DataProvider(**dict_args)

    n_fft, hop_length, num_frame = [dict_args[key] for key in ['n_fft', 'hop_length', 'num_frame']]

    train_dataloader = data_provider.get_train_dataloader(n_fft, hop_length, num_frame)
    valid_dataloader = data_provider.get_valid_dataloader(n_fft, hop_length, num_frame)
#    test_dataloader = data_provider.get_test_dataloader(n_fft, hop_length, num_frame)

    if not dict_args['skip_train']:
        trainer.fit(model, train_dataloader, valid_dataloader)
    # if not dict_args['skip_test']:
    #     trainer.test(model, test_dataloader)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_name', type=str, default='cunet')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints/')
    parser.add_argument('--log_system', type=str, default=True)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--run_id', type=str, default=None)

    parser.add_argument('--skip_train', type=bool, default=False)
    parser.add_argument('--skip_test', type=bool, default=False)


    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "cunet":
        parser = CUNET_Framework.add_model_specific_args(parser)

    parser = DataProvider.add_data_provider_args(parser)
    args = parser.parse_args()

    # train
    main(args)
