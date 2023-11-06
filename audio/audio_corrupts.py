import numpy as np
import torchaudio
import torch

def gaussian_audio(signal, std = 0.02):
    #
    sig, sr = signal
    for i, side in enumerate(sig):
        size = sig.shape
        noise = np.random.normal(loc=0.0, 
                            scale=std, 
                            size=size)
        output = sig + noise
        sig[i] = output
    return sig, sr


def filter_audio(signal, Fs = 44100, central_freq=500, Q=10):
    #
    sig, sr = signal
    # signal_torch = torch.from_numpy(sig)
    for i, side in enumerate(sig):
        output = torchaudio.functional.bandpass_biquad(side, 
                                                    Fs,
                                                    central_freq, 
                                                    Q).numpy()
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
    sig, sr = signal
    for i, side in enumerate(sig):
        window_len= int(len(side)/4)
        signal_slice = signal[:(4*window_len)].copy()
        signal_slice = signal_slice.reshape((4, -1))

        order = np.linspace(0,3,4, dtype='int')
        np.random.shuffle(order)

        signal_copy = np.zeros_like(signal_slice)
        for i, idx in enumerate(order):
            signal_copy[i] = signal_slice[idx]
        output = signal_copy.reshape(len(signal))
        sig[i] = output
    return sig, sr