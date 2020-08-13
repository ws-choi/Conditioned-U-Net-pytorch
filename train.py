from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from models.separation_framework import Magnitude_Masking, ConditionalSS_Framework
import os


def main(args):
    dict_args = vars(args)
    temp_args, _ = parser.parse_known_args()

    model_name = temp_args.model_name
    if temp_args.tensorboard_path is not None:
        if not os.path.exists(temp_args.tensorboard_path):
            os.mkdir(temp_args.tensorboard_path)
        tb_logger = pl_loggers.TensorBoardLogger(temp_args.tensorboard_path, name=model_name)
    else:
        tb_logger = True  # default

    if temp_args.float16:
        trainer = Trainer(gpus=temp_args.gpus, precision=16, logger=tb_logger)
    else:
        trainer = Trainer(gpus=temp_args.gpus, logger=tb_logger)

    model = Magnitude_Masking(**dict_args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = ConditionalSS_Framework.add_model_specific_args(parser)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--tensorboard_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='cunet_dense_simple')

    args = parser.parse_args()

    # train
    main(args)
