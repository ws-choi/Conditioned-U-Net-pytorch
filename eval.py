from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
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

    if dict_args['log_system'] == 'wandb':
        logger = WandbLogger(project='source_separation', tags=model_name, offline=False, id=temp_args.run_id)
        logger.log_hyperparams(model.hparams)
        logger.watch(model, log='all')

    elif dict_args['log_system'] == 'tensorboard':
        if not os.path.exists(temp_args.tensorboard_path):
            os.mkdir(temp_args.tensorboard_path)
        logger = pl_loggers.TensorBoardLogger(temp_args.tensorboard_path, name=model_name)
    else:
        logger = True  # default

    trainer = Trainer(
        gpus=temp_args.gpus,
        logger=logger,
    )

    ckpt_path = dict_args['ckpt_path']
    trainer.test(model, ckpt_path=ckpt_path)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_name', type=str, default='cunet')
    parser.add_argument('--checkpoints_path', type=str, default=None)
    parser.add_argument('--log_system', type=str, default='wandb')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--ckpt_path', type=str, default='best')

    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "cunet":
        parser = CUNET_Framework.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)
