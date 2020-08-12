from argparse import ArgumentParser

from pytorch_lightning import Trainer

from models.separation_framework import Magnitude_Masking, ConditionalSS_Framework


def main(args):
    dict_args = vars(args)

    temp_args, _ = parser.parse_known_args()
    if temp_args.float16:
        trainer = Trainer(gpus=temp_args.gpus, precision=16)
    else:
        trainer = Trainer(gpus=temp_args.gpus)

    model = Magnitude_Masking(**dict_args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = ConditionalSS_Framework.add_model_specific_args(parser)
    parser.add_argument('--float16', type=bool, default=False)

    args = parser.parse_args()

    # train
    main(args)
