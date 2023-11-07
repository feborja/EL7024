import numpy as np
import torchaudio
import torch

def gaussian_audio(signal, std = 0.02):
    #
    sig, sr = signal
    noise = torch.normal(mean = 0.0, std = std, size = sig.shape)
    output = noise + sig
    return output, sr


def filter_audio(signal, central_freq=500, Q=10):
    #
    sig, sr = signal
    # signal_torch = torch.from_numpy(sig)
    for i, side in enumerate(sig):
        output = torchaudio.functional.bandpass_biquad(side, 
                                                    sr,
                                                    central_freq, 
                                                    Q)
        sig[i] = output
    return sig, sr


def silent_audio(signal, window_num = 10, prob = 0.5):
    #
    sig, sr = signal
    signal_window = sig.clone().reshape((2, window_num, -1))
    for wnd in signal_window:
        for i, _ in enumerate(wnd):
            luck = np.random.rand(1).item()
            if luck < prob:
                wnd[i] = torch.zeros_like(wnd[i])
    # print(sig.shape)
    # print(signal_window.shape)
    output = signal_window.reshape((2, sig.shape[1]))
    return output, sr


def random_audio_permute(signal):
    #
    sig, sr = signal
    for i, side in enumerate(sig):
        window_len= int(len(side)/4)
        signal_slice = side[:(4*window_len)].clone()
        signal_slice = signal_slice.reshape((4, -1))

        order = np.linspace(0,3,4, dtype='int')
        np.random.shuffle(order)

        signal_copy = torch.zeros_like(signal_slice)
        for j, idx in enumerate(order):
            signal_copy[j] = signal_slice[idx]
        output = signal_copy.reshape(len(side))
        sig[i] = output
    return sig, sr