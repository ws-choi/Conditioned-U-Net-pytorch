from argparse import ArgumentParser

from pytorch_lightning import Trainer

from models.separation_framework import Magnitude_Masking, ConditionalSS_Framework


def main(args):
    dict_args = vars(args)
    model = Magnitude_Masking(**dict_args)

    temp_args, _ = parser.parse_known_args()
    if(temp_args.float16):
        trainer = Trainer(gpus=1, precision=16)
    else:
        trainer = Trainer(gpus=1)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = ConditionalSS_Framework.add_model_specific_args(parser)
    parser.add_argument('--float16', type=bool, default=False)

    # figure out which model to use
    # . ('--data_in_memory', type=bool, default=False, help='True if len(mem) > 80G')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    # temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    # if temp_args.model_name == 'gan':
    #     parser = GoodGAN.add_model_specific_args(parser)
    # elif temp_args.model_name == 'mnist':
    #     parser = LitMNIST.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)
