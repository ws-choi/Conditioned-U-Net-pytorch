# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):

    # Fourier
    n_fft = setting(default=1024)
    hop_length = setting(default=256)
    num_frame = setting(default=256)

    # lr
    lr = setting(default=1e-3)

