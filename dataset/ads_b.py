import numpy as np
import random
import numpy as np
import math
import json
import random
import util.params as params
from easydict import EasyDict
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import util


def get_num_class_pretraindata():
    x = np.load(f"F:/FS-SEI_4800/Dataset/X_train_90Class.npy") #change it according to your path
    y = np.load(f"F:/FS-SEI_4800/Dataset/Y_train_90Class.npy") #change it according to your path
    x = x.transpose(0, 2, 1)
    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[:]
    return x[train_index_shot], y[train_index_shot]


def get_num_class_finetunedata(k):
    x = np.load(f"F:/FS-SEI_4800/Dataset/X_train_10Class.npy")
    y = np.load(f"F:/FS-SEI_4800/Dataset/Y_train_10Class.npy")
    x = x.transpose(0, 2, 1)
    x_test = np.load(f"F:/FS-SEI_4800/Dataset/X_test_10Class.npy")
    y_test = np.load(f"F:/FS-SEI_4800/Dataset/Y_test_10Class.npy")
    x_test = x_test.transpose(0, 2, 1)
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    return x[finetune_index_shot], x_test, y[finetune_index_shot], y_test


def TrainDataset(num):
    x = np.load(f"F:/FS-SEI_4800/Dataset/X_train_{num}Class.npy")
    y = np.load(f"F:/FS-SEI_4800/Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.1, random_state=30)
    return X_train, X_val, Y_train, Y_val


def PreTrainDataset_prepared(class_index, sample_num, flag_permutate=False):
    X_train_ul, Y_train_ul = get_num_class_pretraindata()
    train_index_shot = []
    for i in class_index:
        index_classi = [index for index, value in enumerate(Y_train_ul) if value == i]
        # train_index_shot += index_classi[0:sample_num]
        train_index_shot += index_classi[:]
    X_train_shot = X_train_ul[train_index_shot]
    Y_train_shot = Y_train_ul[train_index_shot]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_shot, Y_train_shot, test_size=0.2, random_state=30)
    # X_train_shot, Y_train_shot = add_gaussian_noise(X_train, Y_train, 10)
    if flag_permutate:
        augmented_signals, augmented_labels = augment_samples(X_train_shot, Y_train_shot, 4)
        return augmented_signals, augmented_labels.astype(np.uint8)
    else:
        return X_train_shot, Y_train_shot.astype(np.uint8), X_val, Y_val.astype(np.uint8)


def FineTuneDataset_prepared(k):
    X_train, X_test, Y_train, Y_test = get_num_class_finetunedata(k)
    Y_train = Y_train.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)

    # max_value = X_train.max()
    # min_value = X_train.min()
    #
    # X_train = (X_train - min_value) / (max_value - min_value)
    # X_test = (X_test - min_value) / (max_value - min_value)

    return X_train, X_test, Y_train, Y_test


def denormalize_data(data, inverse_para):
    [data_min, data_max] = [inverse_para[:, :, 0], inverse_para[:, :, 1]]
    [data_min, data_max] = [data_min[:, :, np.newaxis], data_max[:, :, np.newaxis]]
    # Min-Max Denormalization
    data_denorm = data * (data_max - data_min) + data_min
    return data_denorm


def norm_data(data):
    # Z-Score Normalization
    # mean = data.mean(axis=(0, 2), keepdims=True)
    # std = data.std(axis=(0, 2), keepdims=True)
    # data_zscore = (data - mean) / std
    #
    # Min-Max Normalization
    if isinstance(data, np.ndarray):
        # for NumPy array
        data_min = np.min(data, axis=(1, 2), keepdims=True)
        data_max = np.max(data, axis=(1, 2), keepdims=True)
        data_minmax = (data - data_min) / (data_max - data_min)
    elif isinstance(data, torch.Tensor):
        # for PyTorch tensor
        data_min = data.min(dim=(1, 2), keepdim=True).values
        data_max = data.max(dim=(1, 2), keepdim=True).values
        data_minmax = (data - data_min) / (data_max - data_min)
    else:
        raise TypeError("Unsupported data type. Must be either numpy.ndarray or torch.Tensor.")

    return data_minmax, np.concatenate((data_min, data_max), axis=2)


def get_dataloader(signal_repre, class_index, sample_num=100, flag_permutate=False):
    print('#----------Preparing for the ADS-B dataset----------#')
    # X_train, Y_train, X_val, Y_val = PreTrainDataset_prepared(class_index, sample_num=sample_num,
    #                                                           flag_permutate=flag_permutate)
    X_train, X_val, Y_train, Y_val = TrainDataset(20)
    ori_X, ori_Y = X_val, Y_val
    data = EasyDict()
    if signal_repre == 'origin':
        X_train, train_norm_par = norm_data(X_train)
        X_val, val_norm_par = norm_data(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
        # data.train_norm_param = train_norm_par
        # data.test_norm_param = val_norm_par
    elif signal_repre == 'dwt':
        X_train_dwt = util.preprocessing.gendwt(X_train)
        X_val_dwt = util.preprocessing.gendwt(X_val)
        X_train_dwt, X_val_dwt = norm_data(X_train_dwt), norm_data(X_val_dwt)
        train_dataset = TensorDataset(torch.Tensor(X_train_dwt), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                                  num_workers=0, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val_dwt), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size,
                                num_workers=0, shuffle=True)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'fft':
        X_train_fft = util.preprocessing.genfft(X_train)
        X_val_fft = util.preprocessing.genfft(X_val)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        X_val_fft, val_norm_par = norm_data(X_val_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_fft), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
        # data.train_norm_param = train_norm_par
        # data.test_norm_param = val_norm_par
    print('load dataset over')
    return data, ori_X, ori_Y


def get_dataloader_new(signal_repre, class_index, sample_num=100, flag_permutate=False, test_flag=False):
    print('#----------Preparing for the ADS-B dataset----------#')
    X_train_ul, Y_train_ul = PreTrainDataset_prepared(class_index, sample_num=sample_num, test_flag=test_flag,
                                                      flag_permutate=flag_permutate)
    if signal_repre == 'origin':
        X_train, train_norm_par = norm_data(X_train_ul)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train_ul))
    elif signal_repre == 'dwt':
        X_train_dwt = util.preprocessing.gendwt(X_train_ul)
        X_train_dwt = norm_data(X_train_dwt)
        train_dataset = TensorDataset(torch.Tensor(X_train_dwt), torch.Tensor(Y_train_ul))
    elif signal_repre == 'fft':
        X_train_fft = util.preprocessing.genfft(X_train_ul)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(Y_train_ul))
    return train_dataset


def augment_samples(signal_samples, labels, num_augmentations=5, block_size=64):
    N, _, total_length = signal_samples.shape
    augmented_signals = []
    augmented_labels = []

    for i in range(N):
        original_signal = signal_samples[i]
        original_label = labels[i]

        # Retain the original signals and labels
        augmented_signals.append(original_signal)
        augmented_labels.append(original_label)

        # For each sample to do num_augmentations times to perform the shuffle operation again and expand
        for _ in range(num_augmentations):
            shuffled_signal = shuffle_signal(original_signal, block_size)
            augmented_signals.append(shuffled_signal)
            augmented_labels.append(original_label)

    # Transfer as numpy array
    augmented_signals = np.array(augmented_signals)
    augmented_labels = np.array(augmented_labels)

    return augmented_signals, augmented_labels


def shuffle_signal(signal, block_size=64):
    # Get the dimension of the signal
    num_rows, total_length = signal.shape
    assert num_rows == 2, "The first dimension of the signal must be 2, presents IQ two ways"

    # Ensure that the signal length can be devided by block_size 
    assert total_length % block_size == 0, "The signal length must be an integer multiple of block_size"

    # Split the signal according to block_size
    num_blocks = total_length // block_size
    reshaped_signal = signal.reshape(num_rows, num_blocks, block_size)

    # Generate random shuffled indices
    shuffle_indices = np.random.permutation(num_blocks)

    # Scramble the I channel and Q channel in the same order
    reshaped_signal = reshaped_signal[:, shuffle_indices, :]

    # Re-splice the signal
    shuffled_signal = reshaped_signal.reshape(num_rows, total_length)

    return shuffled_signal


def add_gaussian_noise(signal_matrix, labels, snr_db):
    """
    Adds Gaussian noise to each sample in the signal matrix according to the specified SNR in dB.

    Parameters:
        signal_matrix (np.array): The input signal matrix of shape (N, 2, 4800).
        labels (np.array): The label array of shape (N,).
        snr_db (float): The desired Signal-to-Noise Ratio in decibels.

    Returns:
        np.array: The expanded signal matrix including original and noisy samples.
        np.array: The expanded label array including labels for noisy samples.
    """
    # Calculate SNR ratio from dB
    snr_ratio = 10 ** (snr_db / 10)

    # Initialize lists to hold new samples and labels
    new_samples = []
    new_labels = []

    # Iterate through each sample
    for idx, sample in enumerate(signal_matrix):
        # Compute the signal power
        signal_power = np.mean(sample ** 2)

        # Calculate noise power based on signal power and SNR
        noise_power = signal_power / snr_ratio

        # Standard deviation of the noise
        noise_std = np.sqrt(noise_power)

        # Add the original sample first
        new_samples.append(sample)
        new_labels.append(labels[idx])

        # Generate 4 noisy samples per original sample
        for _ in range(4):
            noise = noise_std * np.random.randn(*sample.shape)
            noisy_sample = sample + noise
            new_samples.append(noisy_sample)
            new_labels.append(labels[idx])

    # Convert lists to numpy arrays
    expanded_signal_matrix = np.array(new_samples)
    expanded_labels = np.array(new_labels)

    return expanded_signal_matrix, expanded_labels


def trans_gaussian_noise(signal_matrix, labels, snr_db):
    # Calculate SNR ratio from dB
    snr_ratio = 10 ** (snr_db / 10)

    # Initialize lists to hold new samples and labels
    new_samples = []
    new_labels = []

    # Iterate through each sample
    for idx, sample in enumerate(signal_matrix):
        # Compute the signal power
        signal_power = np.mean(sample ** 2)

        # Calculate noise power based on signal power and SNR
        noise_power = signal_power / snr_ratio

        # Standard deviation of the noise
        noise_std = np.sqrt(noise_power)

        noise = noise_std * np.random.randn(*sample.shape)
        noisy_sample = sample + noise
        new_samples.append(noisy_sample)
        new_labels.append(labels[idx])

    # Convert lists to numpy arrays
    expanded_signal_matrix = np.array(new_samples)
    expanded_labels = np.array(new_labels)

    return expanded_signal_matrix, expanded_labels


if __name__ == "__main__":
    # data = get_dataloader('origin')
    # Example usage
    N = 10  # number of samples
    get_num_class_finetunedata(100)
    signal_matrix = np.random.randn(N, 2, 4800)  # example signal matrix
    labels = np.random.randint(0, 5, size=(N, 1))  # example labels for each sample
    # noisy_signal_matrix, expanded_labels = add_gaussian_noise(signal_matrix, labels, 10)
    noisy_signal_matrix, expanded_labels = trans_gaussian_noise(signal_matrix, labels, 10)
    print(noisy_signal_matrix.shape)  # Expected shape: (N*4, 2, 4800)
    print(expanded_labels.shape)  # Expected shape: (N*4, 1)
