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

    checkpoint_path =dict_args['checkpoints_path']
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

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
        patience=15,
        verbose=False

    )
    if dict_args['float16']:
        trainer = Trainer(
            gpus=temp_args.gpus,
            precision=16,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback
        )
    else:
        trainer = Trainer(
            gpus=temp_args.gpus,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback
        )

    trainer.fit(model)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_name', type=str, default='cunet')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints/')
    parser.add_argument('--log_system', type=str, default='wandb')
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--run_id', type=str, default=None)


    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "cunet":
        parser = CUNET_Framework.add_model_specific_args(parser)


    args = parser.parse_args()

    # train
    main(args)
