import numpy as np
import museval as eval4


def median_nan(a):
    return np.median(a[~np.isnan(a)])


def musdb_all(ref, est, sr):
    '''return sdr, isr, sir, sar'''
    sdr, isr, sir, sar, _ = eval4.metrics.bss_eval(ref, est, window=sr, hop=sr)
    return [median_nan(metric[0]) for metric in [sdr, isr, sir, sar]]
