import numpy as np
import torchaudio
import torch

def gaussian_audio(signal, std = 0.05, idx = 0):
    #
    sig, sr = signal
    noise = torch.normal(mean = 0.0, std = std, size = sig.shape, generator = torch.Generator().manual_seed(int(idx * 1000000)))
    output = noise + sig
    return output, sr


def filter_audio(signal, central_freq=500, Q=10, idx = 0):
    #
    sig, sr = signal
    # signal_torch = torch.from_numpy(sig)
    # for i, side in enumerate(sig):
    output = torchaudio.functional.bandpass_biquad(sig, 
                                                sr,
                                                central_freq, 
                                                Q)
        # sig[i] = output
    return output, sr


def silent_audio(signal, window_num = 5, prob = 0.5, idx = 0):
    #
    sig, sr = signal
    #
    if not (0 <= prob <= 1):
        raise ValueError("Probability should be between 0 and 1")
    # Calculate the number of samples per window
    samples_per_window = sig.shape[1] // window_num
    lucks = torch.rand(size = (window_num,), generator = torch.Generator().manual_seed(int(idx * 1000000)))
    for i in range(window_num):
        luck = lucks[i].item()
        if luck < prob:
            start = i * samples_per_window
            end = (i + 1) * samples_per_window
            sig[:, start:end] = 0

    return sig, sr


def random_audio_permute(signal, idx = 0):
    #
    sig, sr = signal
    for i, side in enumerate(sig):
        window_len= int(len(side)/4)
        signal_slice = side[:(4*window_len)].clone()
        signal_slice = signal_slice.reshape((4, -1))
        #
        rng = np.random.default_rng(int(idx * 1000000))
        order = np.linspace(0,3,4, dtype='int')
        rng.shuffle(order)
        #
        signal_copy = torch.zeros_like(signal_slice)
        for j, idx in enumerate(order):
            signal_copy[j] = signal_slice[idx]
        output = signal_copy.reshape(len(side))
        sig[i] = output
    return sig, sr