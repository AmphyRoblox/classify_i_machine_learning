import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from scipy.signal import stft

# Global variable definition
sampling_rate = 3.5e6
data_length = 26000


def get_cwt(data):
    wavename = 'morl'
    totalscal = 94
    fc = pywt.central_frequency(wavename)
    cparam = 10 * fc * totalscal
    scales = cparam / np.arange(totalscal, 0, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    return [cwtmatr, frequencies]


def wavelet_denoising(data, wavelet='db1', level=3):
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(data, wavelet, mode="per", level=level)
    sigma = (1 / 0.6745) * mad(coeff[-level])

    # Calculate the threshold
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # Apply threshold to the detail coefficients (not the approximation coefficients)
    coeff[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:])

    # Reconstruct the signal
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal


def mad(data):
    return np.median(np.abs(data - np.median(data)))


def gendwt(signals_nt, wavelet='db1', mode='soft', thresh_scale=0.5, out_type='numpy'):
    if torch.is_tensor(signals_nt):
        signals = signals_nt.detach().numpy()
    else:
        signals = signals_nt
    row = signals.shape[0]
    res = np.zeros([row, 2, signals.shape[2]])

    for i in range(row):
        for j in range(2):
            res[i, j, :] = wavelet_denoising(np.squeeze(signals[i, j, :]))

    if out_type == 'tensor':
        res = torch.from_numpy(res).float()
    return res


def genfft(signals_nt, out_type='numpy', inputdim=1):
    if torch.is_tensor(signals_nt):
        signals = signals_nt.detach().numpy()
    else:
        signals = signals_nt
    row = signals.shape[0]

    # Initialize the result array to store the FFT of the two-channel signals
    if inputdim == 2:
        column = signals.shape[3]
        res = np.zeros([row, 1, 2, column])
        for i in range(row):
            complex_signal = np.squeeze(signals[i, 0, 0, :]) + 1j * np.squeeze(signals[i, 0, 1, :])
            # Calculate the FFT of the I-channel signal
            res[i, 0, 0, :] = np.real(np.fft.fft(complex_signal))
            # Calculate the FFT of the Q-channel signal
            res[i, 0, 1, :] = np.imag(np.fft.fft(complex_signal))
    else:
        column = signals.shape[2]
        res = np.zeros([row, 2, column])
        for i in range(row):
            complex_signal = np.squeeze(signals[i, 0, :]) + 1j * np.squeeze(signals[i, 1, :])
            # Calculate the FFT of the I-channel signal
            res[i, 0, :] = np.real(np.fft.fft(complex_signal))
            # Calculate the FFT of the Q-channel signal
            res[i, 1, :] = np.imag(np.fft.fft(complex_signal))

    if out_type == 'tensor':
        # Convert the result back to a torch tensor, and retain the complex data type.
        res = torch.from_numpy(res).to(torch.float)
    return res


def genifft(fft_data, out_type='numpy'):
    if torch.is_tensor(fft_data):
        fft_data = fft_data.numpy()  # If the input is a torch tensor, convert it to a numpy array for processing

    row = fft_data.shape[0]
    inputdim = len(fft_data.shape)  # Determine the processing method based on the dimensions of the input data

    # Process dual-channel input
    if inputdim == 4:
        column = fft_data.shape[3]
        signals = np.zeros([row, 1, 2, column], dtype=complex)
        for i in range(row):
            # Combine the real part and the imaginary part into a complex signal
            complex_signal = fft_data[i, 0, 0, :] + 1j * fft_data[i, 0, 1, :]
            # Perform inverse FFT
            signals[i, 0, 0, :] = np.real(np.fft.ifft(complex_signal))
            signals[i, 0, 1, :] = np.imag(np.fft.ifft(complex_signal))

    # Process single-channel input
    else:
        column = fft_data.shape[2]
        signals = np.zeros([row, 2, column], dtype=complex)
        for i in range(row):
            # Combine the real part and the imaginary part into a complex signal
            complex_signal = fft_data[i, 0, :] + 1j * fft_data[i, 1, :]
            # Perform inverse FFT
            signals[i, 0, :] = np.real(np.fft.ifft(complex_signal))
            signals[i, 1, :] = np.imag(np.fft.ifft(complex_signal))

    # Convert the result according to the output type
    if out_type == 'tensor':
        signals = torch.from_numpy(signals).to(torch.float)

    return signals


def time2tf(signals, method):
    row = signals.shape[0]
    sample_ = signals[0]
    if method == 'cwt':
        cwt_matrix, freq_ = get_cwt(sample_[0, :])
        # plot_time_freq(np.arange(0,sample_.shape[1],1.0),freq_,cwt_matrix)
        res = np.zeros(row, 2, cwt_matrix.shape[0], cwt_matrix.shape[1])
        for i in range(row):
            res[i, 0, :, :] = get_cwt(signals[i, 0, :])  # Calculate the cwt of the I-th path
            res[i, 1, :, :] = get_cwt(signals[i, 1, :])  # Calculate the cwt of the Q-th path
    elif method == 'stft':
        freqs, t, coef = stft(sample_[0, :], fs=sampling_rate, window='hann', nperseg=1024, noverlap=512, nfft=1024)
        plot_time_freq(t, freqs, coef)
    return res


def get_time_freq(data, method='cwt'):
    if method == 'cwt':
        [coef, freqs] = get_cwt(data)
        return [np.arange(0, len(data), 1.0), freqs, coef]
    elif method == 'stft':
        freqs, t, coef = stft(data, fs=sampling_rate, window='hann', nperseg=1024, noverlap=512, nfft=1024)
        return [t, freqs, coef]


def plot_time_freq(t, freqs, coef):
    plt.figure(figsize=(4, 4))
    plt.contourf(t, freqs, abs(coef))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()
