from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from models.separation_framework import Magnitude_Masking, ConditionalSS_Framework
from pytorch_lightning.loggers import WandbLogger
import wandb
import os


def main(args):
    dict_args = vars(args)
    temp_args, _ = parser.parse_known_args()

    model_name = temp_args.model_name

    from pytorch_lightning.callbacks import ModelCheckpoint

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=10,
        verbose=True,
        monitor='val_loss',
        prefix=''
    )

    model = Magnitude_Masking(**dict_args)

    if temp_args.log_system == 'wandb':
        # if not os.path.exists(temp_args.tensorboard_path):
        #     os.mkdir(temp_args.tensorboard_path)
        # tb_logger = pl_loggers.TensorBoardLogger(temp_args.tensorboard_path, name=model_name)
        logger = WandbLogger(project='source_separation', tags=model_name, offline=False, id='cunets')
        logger.log_hyperparams(model.hparams)
        logger.watch(model, log='all')
    else:
        logger = True  # default

    if temp_args.float16:
        trainer = Trainer(gpus=temp_args.gpus, precision=16, logger=logger, checkpoint_callback=checkpoint_callback)
    else:
        trainer = Trainer(gpus=temp_args.gpus, logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = ConditionalSS_Framework.add_model_specific_args(parser)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--log_system', type=str, default='wandb')
    parser.add_argument('--model_name', type=str, default='cunet_dense_simple')

    args = parser.parse_args()

    # train
    main(args)
