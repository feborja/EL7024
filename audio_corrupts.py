import numpy as np
import torchaudio
import torch

def gaussian_audio(signal, std = 0.02):
    size = signal.shape
    noise = np.random.normal(loc=0.0, 
                         scale=std, 
                         size=size)
    output = signal + noise
    return output


def filter_audio(signal, Fs = 44100, central_freq=500, Q=10):
    signal_torch = torch.from_numpy(signal)
    output = torchaudio.functional.bandpass_biquad(signal_torch, 
                                                   Fs,
                                                   central_freq, 
                                                   Q).numpy()
    return output


def silent_audio(signal, window_num = 10, prob = 0.5):
    signal_window = signal.reshape((window_num, -1)).copy()

    for i, _ in enumerate(signal_window):
        luck = np.random.rand(1).item()
        if luck < prob:
            signal_window[i] = np.zeros_like(signal_window[i])
    output = signal_window.reshape(len(signal))
    return output


def random_audio_permute(signal):
    window_len= int(len(signal)/4)
    signal_slice = signal[:(4*window_len)].copy()
    signal_slice = signal_slice.reshape((4, -1))

    order = np.linspace(0,3,4, dtype='int')
    np.random.shuffle(order)

    signal_copy = np.zeros_like(signal_slice)
    for i, idx in enumerate(order):
        signal_copy[i] = signal_slice[idx]
    output = signal_copy.reshape(len(signal))
    return output