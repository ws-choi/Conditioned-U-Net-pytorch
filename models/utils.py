from models.separation_framework import CUNET_Framework


def get_model(model_name, dict_args):
    if model_name == 'cunet':
        return CUNET_Framework(**dict_args)
    else:
        raise NotImplementedError
