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
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_dataloader(signal_repre, class_index, sample_num=400):
    print('#----------Preparing for the wifi dataset----------#')
    X_train_ul, Y_train_ul = PreTrainDataset_prepared(62, classi=class_index, sample_num=sample_num)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_ul, Y_train_ul, test_size=0.2, random_state=30)
    data = EasyDict()
    if signal_repre == 'origin':
        X_train, train_norm_par = norm_data(X_train)
        X_val, val_norm_par = norm_data(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
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
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val_dwt), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=True)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'fft':
        X_train_fft = util.preprocessing.genfft(X_train)
        X_val_fft = util.preprocessing.genfft(X_val)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        X_val_fft, val_norm_par = norm_data(X_val_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_fft), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
        # data.train_norm_param = train_norm_par
        # data.test_norm_param = val_norm_par
    return train_dataset, data


def get_dataloader_new(signal_repre, class_index, sample_num=100, flag_permutate=False, test_flag=False):
    print('#----------Preparing for the wifi dataset----------#')
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


def WiFi_Dataset_slice(ft, classi):
    devicename = ['3123D7B', '3123D7D', '3123D7E', '3123D52', '3123D54', '3123D58', '3123D64', '3123D65',
                  '3123D70', '3123D76', '3123D78', '3123D79', '3123D80', '3123D89', '3123EFE', '3124E4A']
    data_IQ_wifi_all = np.zeros((1, 2, 4800))
    data_target_all = np.zeros((1,))
    target = 0
    for classes in classi:
        target = classes
        for recoder in range(1):
            inputFilename = f'F:/bishe/KRI-16Devices-RawData/{ft}ft/WiFi_air_X310_{devicename[classes]}_{ft}ft_run{recoder + 1}'
            with open("{}.sigmf-meta".format(inputFilename), 'rb') as read_file:
                meta_dict = json.load(read_file)
            with open("{}.sigmf-data".format(inputFilename), 'rb') as read_file:
                binary_data = read_file.read()
            fullVect = np.frombuffer(binary_data, dtype=np.complex128)
            even = np.real(fullVect)
            odd = np.imag(fullVect)
            length = 4800
            num = 0
            data_IQ_wifi = np.zeros((math.floor(len(even) / length), 2, 4800))
            data_target = np.zeros((math.floor(len(even) / length),))
            for begin in range(0, len(even) - (len(even) - math.floor(len(even) / length) * length), length):
                data_IQ_wifi[num, 0, :] = even[begin:begin + length]
                data_IQ_wifi[num, 1, :] = odd[begin:begin + length]
                data_target[num,] = target
                num = num + 1
            data_IQ_wifi_all = np.concatenate((data_IQ_wifi_all, data_IQ_wifi), axis=0)
            data_target_all = np.concatenate((data_target_all, data_target), axis=0)
    return data_IQ_wifi_all[1:, ], data_target_all[1:, ]


def PreTrainDataset_prepared(ft, classi, sample_num, test_flag=False, flag_permutate=False):
    x, y = WiFi_Dataset_slice(ft, classi)
    train_index_shot = []
    for i in classi:
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:sample_num]

    X_train_shot = x[train_index_shot]
    Y_train_shot = y[train_index_shot]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_shot, Y_train_shot, test_size=0.2, random_state=30)
    X_train, Y_train = add_gaussian_noise(X_train, Y_train, 10)
    if flag_permutate:
        augmented_signals, augmented_labels = augment_samples(X_train_shot, Y_train_shot, 4)
        return augmented_signals, augmented_labels.astype(np.uint8)
    else:
        return X_train, Y_train.astype(np.uint8), X_val, Y_val.astype(np.uint8)


def PreTrainDataset_prepared_snr(ft, classi, sample_num, test_flag=False, flag_permutate=False, snr=None):
    x, y = WiFi_Dataset_slice(ft, classi)
    train_index_shot = []
    for i in classi:
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:sample_num]

    X_train_shot = x[train_index_shot]
    Y_train_shot = y[train_index_shot]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_shot, Y_train_shot, test_size=0.2, random_state=30)
    X_train, Y_train = add_gaussian_noise(X_train, Y_train, snr, trans_flag=True)
    X_val, Y_val = add_gaussian_noise(X_val, Y_val, snr, trans_flag=True, trans_num=1)
    return X_train, Y_train.astype(np.uint8), X_val, Y_val.astype(np.uint8)


def augment_samples(signal_samples, labels, num_augmentations=5, block_size=64):
    N, _, total_length = signal_samples.shape
    augmented_signals = []
    augmented_labels = []

    for i in range(N):
        original_signal = signal_samples[i]
        original_label = labels[i]

        # Keep the original signals and labels
        augmented_signals.append(original_signal)
        augmented_labels.append(original_label)

        # Perform num_augmentations shuffling operations on each sample and augment them
        for _ in range(num_augmentations):
            shuffled_signal = shuffle_signal(original_signal, block_size)
            augmented_signals.append(shuffled_signal)
            augmented_labels.append(original_label)

    # Change to a numpy array
    augmented_signals = np.array(augmented_signals)
    augmented_labels = np.array(augmented_labels)

    return augmented_signals, augmented_labels


def shuffle_signal(signal, block_size=64):
    # Get the dimension of the signal
    num_rows, total_length = signal.shape
    assert num_rows == 2, "The first dimension of the number must be 2, representing the two IQ channels."

    # Ensure that the signal length is divisible by block_size
    assert total_length % block_size == 0, "The signal length must be an integer multiple of block_size"

    # Split the signal according to block_size
    num_blocks = total_length // block_size
    reshaped_signal = signal.reshape(num_rows, num_blocks, block_size)

    # 生成随机的打乱索引Generate random shuffled indices
    shuffle_indices = np.random.permutation(num_blocks)

    # Scramble the I channel and Q channel in the same order
    reshaped_signal = reshaped_signal[:, shuffle_indices, :]

    # Re-splice the signal
    shuffled_signal = reshaped_signal.reshape(num_rows, total_length)

    return shuffled_signal


def add_gaussian_noise(signal_matrix, labels, snr_db, trans_flag=False, trans_num=1):
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

        if not trans_flag:
            new_samples.append(sample)
            new_labels.append(labels[idx])

        for _ in range(trans_num):
            noise = noise_std * np.random.randn(*sample.shape)
            noisy_sample = sample + noise
            new_samples.append(noisy_sample)
            new_labels.append(labels[idx])

    # Convert lists to numpy arrays
    expanded_signal_matrix = np.array(new_samples)
    expanded_labels = np.array(new_labels)

    return expanded_signal_matrix, expanded_labels


def reshape2square(x, target_shape):
    target_length = np.prod(target_shape)  # 98*98 = 9604

    # Initialize the result array
    result = np.empty((x.shape[0], 1, *target_shape))

    # Iterate through the array and process each element
    for i in range(x.shape[0]):
        # Flatten the current element
        flat_element = x[i].flatten()

        # Calculate the number of elements that need to be filled in
        padding_length = target_length - flat_element.size

        # Fill and reshape into a square matrix
        padded_element = np.pad(flat_element, (0, padding_length), 'constant', constant_values=(0,))
        square_matrix = padded_element.reshape(target_shape)

        # Store into the result array
        result[i] = square_matrix
    return result


def FineTuneDataset_prepared(ft, classi, k):
    x, y = WiFi_Dataset_slice(ft, classi)
    test_index_shot = []
    finetune_index_shot = []
    for i in classi:
        i -= classi[0]
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi[0:1000], k)
        test_index_shot += index_classi[1000:2000]

    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]

    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)
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
        # For NumPy array
        data_min = np.min(data, axis=(1, 2), keepdims=True)
        data_max = np.max(data, axis=(1, 2), keepdims=True)
        data_minmax = (data - data_min) / (data_max - data_min)
    elif isinstance(data, torch.Tensor):
        # For PyTorch tensor
        data_min = data.min(dim=(1, 2), keepdim=True).values
        data_max = data.max(dim=(1, 2), keepdim=True).values
        data_minmax = (data - data_min) / (data_max - data_min)
    else:
        raise TypeError("Unsupported data type. Must be either numpy.ndarray or torch.Tensor.")

    return data_minmax, np.concatenate((data_min, data_max), axis=2)


def get_dataloader(signal_repre, class_index, sample_num=100, flag_permutate=False):
    print('#----------Preparing for the wifi dataset----------#')
    X_train, Y_train, X_val, Y_val = PreTrainDataset_prepared(ft=62, classi=class_index, sample_num=sample_num,
                                                              flag_permutate=flag_permutate)
    ori_X, ori_Y = X_val, Y_val
    data = EasyDict()
    if signal_repre == 'origin':
        X_train, train_norm_par = norm_data(X_train)
        X_val, val_norm_par = norm_data(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
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
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val_dwt), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=True)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'fft':
        X_train_fft = util.preprocessing.genfft(X_train)
        X_val_fft = util.preprocessing.genfft(X_val)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        X_val_fft, val_norm_par = norm_data(X_val_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_fft), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
        # data.train_norm_param = train_norm_par
        # data.test_norm_param = val_norm_par
    return data, ori_X, ori_Y


def get_dataloader_snr(signal_repre, class_index, sample_num=100, flag_permutate=False, snr=None):
    print('#----------Preparing for the wifi dataset----------#')
    X_train, Y_train, X_val, Y_val = PreTrainDataset_prepared_snr(ft=62, classi=class_index,
                                                                  sample_num=params.sample_num,
                                                                  flag_permutate=flag_permutate, snr=snr)
    ori_X, ori_Y = X_val, Y_val
    data = EasyDict()
    if signal_repre == 'origin':
        X_train, train_norm_par = norm_data(X_train)
        X_val, val_norm_par = norm_data(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
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
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val_dwt), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=True)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'fft':
        X_train_fft = util.preprocessing.genfft(X_train)
        X_val_fft = util.preprocessing.genfft(X_val)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        X_val_fft, val_norm_par = norm_data(X_val_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_fft), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'vmd':
        mat_data = loadmat('./train_xy_denoise.mat', )
        X_train = mat_data['X_train_denoise']  # Replace according to the actual variable names in your .mat file 'data'
        Y_train = mat_data['Y_train'].T
        mat_data = loadmat('./val_xy_denoise.mat', )
        X_val = mat_data['X_val_denoise']  # Replace according to the actual variable names in your .mat file 'data'
        Y_val = mat_data['Y_val'].T
        X_train, train_norm_par = norm_data(X_train)
        X_val, val_norm_par = norm_data(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
    return data, ori_X, ori_Y


def get_dataloader_new(signal_repre, class_index, sample_num=100, flag_permutate=False, test_flag=False):
    print('#----------Preparing for the wifi dataset----------#')
    X_train_ul, Y_train_ul = PreTrainDataset_prepared(62, class_index, sample_num=sample_num, test_flag=test_flag,
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


if __name__ == '__main__':
    # get_dataloader_new('origin', 10)
    X_train_ul, Y_train_ul, _, _ = PreTrainDataset_prepared_snr(ft=62, classi=np.arange(0, 16), sample_num=100,
                                                                flag_permutate=False, snr=-4)
    # # test_data=np.random.rand(10,2,10)
    # norm_test, para = norm_data(X_train_ul)
    # invers_data = denormalize_data(norm_test, para)
    # print(invers_data)
