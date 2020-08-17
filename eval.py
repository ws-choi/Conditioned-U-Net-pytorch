from argparse import ArgumentParser
from warnings import warn

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

    if dict_args['gpus'] > 1 :#or dict_args['num_workers'] > 1:
        # warn('# gpu and num_workers should be 1, Not implemented: museval for distributed parallel')
        dict_args['gpus'] = 1
        # dict_args['num_workers'] = 1
        #
        # distributed_backend = 'ddp'

    if model_name == 'cunet':
        model = CUNET_Framework(**dict_args)
    else:
        raise NotImplementedError

    if dict_args['log_system'] == 'wandb':
        logger = WandbLogger(project='source_separation', tags=model_name, offline=False, id=dict_args['run_id'])
        logger.log_hyperparams(model.hparams)
        # logger.watch(model, log='all')

    elif dict_args['log_system'] == 'tensorboard':
        if not os.path.exists(temp_args.tensorboard_path):
            os.mkdir(temp_args.tensorboard_path)
        logger = pl_loggers.TensorBoardLogger(temp_args.tensorboard_path, name=model_name)
    else:
        logger = True  # default

    ckpt_path = dict_args['ckpt_path']

    assert (ckpt_path is not None)
    model = model.load_from_checkpoint(ckpt_path)

    data_provider = DataProvider(**dict_args)

    n_fft, hop_length, num_frame = [dict_args[key] for key in ['n_fft', 'hop_length', 'num_frame']]

    test_dataloader = data_provider.get_test_dataloader(n_fft, hop_length, num_frame)

    trainer = Trainer(
        gpus=dict_args['gpus'],
        logger=logger,
        resume_from_checkpoint=ckpt_path
        # distributed_backend=distributed_backend
    )

    trainer.test(model, test_dataloader)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_name', type=str, default='cunet')
    parser.add_argument('--checkpoints_path', type=str, default=None)
    parser.add_argument('--log_system', type=str, default='wandb')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)

    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "cunet":
        parser = CUNET_Framework.add_model_specific_args(parser)

    parser = DataProvider.add_data_provider_args(parser)
    args = parser.parse_args()

    # train
    main(args)
