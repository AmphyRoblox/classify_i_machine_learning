import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from easydict import EasyDict
from util.preprocessing import genfft, gendwt
import util.params as params


def norm_data(data, method='minmax'):
    # Z-Score Normalization
    # mean = data.mean(axis=(0, 2), keepdims=True)
    # std = data.std(axis=(0, 2), keepdims=True)
    # data_zscore = (data - mean) / std
    if method == 'zscore':
        if isinstance(data, np.ndarray):
            # For NumPy Array
            mean = data.mean(axis=(1, 2), keepdims=True)
            std = data.std(axis=(1, 2), keepdims=True)
            data_zscore = (data - mean) / std
        elif isinstance(data, torch.Tensor):
            # For PyTorch tensor
            mean = data.mean(dim=(1, 2), keepdim=True).values
            std = data.std(dim=(1, 2), keepdim=True).values
            data_zscore = (data - mean) / std
        else:
            raise TypeError("Unsupported data type. Must be either numpy.ndarray or torch.Tensor.")
        return data_zscore
    elif method == 'minmax':
        # Min-Max Normalization
        if isinstance(data, np.ndarray):
            # For NumPy Array
            data_min = np.min(data, axis=2, keepdims=True)
            data_max = np.max(data, axis=2, keepdims=True)
            data_minmax = (data - data_min) / (data_max - data_min)
        elif isinstance(data, torch.Tensor):
            # For PyTorch tensor
            data_min = data.min(dim=2, keepdim=True).values
            data_max = data.max(dim=2, keepdim=True).values
            data_minmax = (data - data_min) / (data_max - data_min)
        else:
            raise TypeError("Unsupported data type. Must be either numpy.ndarray or torch.Tensor.")
        return data_minmax
    else:
        raise TypeError("Unsupported normalization method.")
    

def get_dataloader(signal_repre, directory_path):
    print('#----------Preparing for the dataset----------#')
    X_train, Y_train, X_val, Y_val, test_files = read_and_process_signals(directory_path)
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train_ul, Y_train_ul, test_size=0.2, random_state=30)
    data = EasyDict()
    if signal_repre == 'origin':
        X_train = norm_data(X_train)
        X_val = norm_data(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'dwt':
        X_train_dwt = gendwt(X_train)
        X_val_dwt = gendwt(X_val)
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
        X_train_fft = genfft(X_train)
        X_val_fft = genfft(X_val)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        X_val_fft, val_norm_par = norm_data(X_val_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(Y_train), torch.Tensor(train_norm_par))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_fft), torch.Tensor(Y_val), torch.Tensor(val_norm_par))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
    return data


def add_noise(data, snr_db, add_int=True):
    """ Adds Gaussian white noise to a signal based on the SNR in dB. """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate signal power and required noise power for given SNR
    signal_power = np.mean(data ** 2)
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)

    # Add noise to the original signal
    noisy_data = data + np.round(noise).astype(np.int64)
    return noisy_data


def process_files(files, data_list, label_list, directory, points_per_sample, overlap, time_mask_flag=False,
                  time_shift_flag=False, num_noises_per_sample=5, num_shifted=3):
    all_data = []
    file_list = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        data = np.fromfile(filepath, dtype=np.int16).astype(np.int64)
        start_index = 0
        total_points = data.size
        num_samples = 0
        all_data.append(data)
        file_list.append(filename)
        while start_index + points_per_sample <= total_points:
            end_index = start_index + points_per_sample
            I = data[start_index:end_index:2]
            Q = data[start_index + 1:end_index:2]
            reshaped_data = np.vstack((I, Q))

            # First, add the original data sample to the list
            data_list.append(reshaped_data.copy())
            num_samples += 1

            # Then add noise multiple times to the same sample
            for _ in range(num_noises_per_sample):
                random_snr = np.random.uniform(0, 20)
                reshaped_data_noisy = add_noise(reshaped_data, random_snr)
                data_list.append(reshaped_data_noisy)

            # Update label list: 1 label for the original and `num_noises_per_sample` for the noisy versions
            label_part = filename.split('Dev')[1]
            label = int(''.join(filter(str.isdigit, label_part.split('_')[0]))) - 1
            label_list.extend([label] * (num_noises_per_sample + 1))  # Extend the list with repeated labels
            if time_mask_flag:
                data_time_mask = time_mask(reshaped_data, 1, 500)
                data_list.append(data_time_mask)
                label_list.extend([label])
            if time_shift_flag:
                for _ in range(num_shifted):
                    data_shifted = random_shift(reshaped_data)
                    data_list.append(data_shifted)
                label_list.extend([label] * num_shifted)

            start_index += overlap

    return all_data, file_list


def read_and_process_signals(directory, points_per_sample=10000, overlap=3000, test_size=0.2, time_shift_flag=False):
    # Create a dictionary to hold files for each device
    device_files = {}

    # Step 1: Organize files by device
    for filename in os.listdir(directory):
        if filename.endswith(".dat"):
            # Extract device label from filename
            device_label = filename.split('Dev')[1].split('_')[0]
            if device_label not in device_files:
                device_files[device_label] = []
            device_files[device_label].append(filename)

    # Step 2: Split files for each device into train and test sets
    train_files = []
    test_files = []
    seed_value = params.data_seed
    np.random.seed(seed_value)
    random_states = np.random.randint(0, 10000, size=10)
    for i, (device_label, files) in enumerate(device_files.items()):
        train, test = train_test_split(files, test_size=test_size, random_state=random_states[i])
        train_files.extend(train)
        test_files.extend(test)

    # Process training files
    train_data = []
    train_labels = []
    process_files(train_files, train_data, train_labels, directory, points_per_sample, overlap, time_mask_flag=False,
                  time_shift_flag=time_shift_flag)

    # Process testing files
    test_data = []
    test_labels = []
    process_files(test_files, test_data, test_labels, directory, points_per_sample, overlap,
                  num_noises_per_sample=0)

   
    final_train_data = np.stack(train_data, axis=0)  # Reshape to (num_files*num_samples) x 2 x 5000
    final_train_labels = np.array(train_labels)
    final_test_data = np.stack(test_data, axis=0)
    final_test_labels = np.array(test_labels)

    return final_train_data, final_train_labels, final_test_data, final_test_labels, test_files


def random_shift(signal):
    """
    Apply a random cyclic shift to the signal.

    Parameters:
    signal (np.ndarray): Input signal of shape (num_samples, num_channels, signal_length)

    Returns:
    np.ndarray: Shifted signal.
    """
    num_channels, signal_length = signal.shape
    shifted_signal = np.empty_like(signal)
    shift = np.random.randint(signal_length)  # Randomly generate a translation amount for each sample
    for j in range(num_channels):
        shifted_signal[j] = np.roll(signal[j], shift)  # All channels use the same translation amount

    return shifted_signal


def time_mask(signal, num_masks=1, mask_length=500):
    """
    Apply time masking to the signal.

    Parameters:
    signal (np.ndarray): Input signal of shape (num_samples, num_channels, signal_length)
    num_masks (int): Number of masks to apply.
    mask_length (int): Length of each mask.

    Returns:
    np.ndarray: Masked signal.
    """
    num_samples, signal_length = signal.shape
    masked_signal = signal.copy()

    for i in range(num_samples):
        for _ in range(num_masks):
            t = np.random.randint(0, signal_length - mask_length)
            masked_signal[i, t:t + mask_length] = 0  # or other mask values, such as np.nan

    return masked_signal


def process_files_test(files, data_list, label_list, directory, points_per_sample, overlap, num_noises_per_sample=0):
    all_data = []
    file_list = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        data = np.fromfile(filepath, dtype=np.int16).astype(np.int64)
        start_index = 0
        total_points = data.size
        num_samples = 0
        all_data.append(data)
        file_list.append(filename)
        while start_index + points_per_sample <= total_points:
            end_index = start_index + points_per_sample
            I = data[start_index:end_index:2]
            Q = data[start_index + 1:end_index:2]
            reshaped_data = np.vstack((I, Q))

            # First, add the original data sample to the list
            data_list.append(reshaped_data.copy())
            num_samples += 1

            # # Then add noise multiple times to the same sample
            # for _ in range(num_noises_per_sample):
            #     random_snr = np.random.uniform(0, 20)
            #     reshaped_data_noisy = add_noise(reshaped_data, random_snr)
            #     data_list.append(reshaped_data_noisy)

            # # Update label list: 1 label for the original and `num_noises_per_sample` for the noisy versions
            label_part = filename.split('_')[-1]  # Get the last part after the underscore
            label = int(label_part.split('.')[0])  # Remove the file extension and convert to int
            label_list.extend([label] * (num_noises_per_sample + 1))  # Extend the list with repeated labels

            start_index += overlap

    return all_data, file_list


def read_and_process_signals_test(directory, points_per_sample=10000, overlap=5000, test_size=0.2):
    # Step 1: Collect all files
    all_files = [f for f in os.listdir(directory) if f.endswith(".dat")]
    all_files_train = all_files[0:400]
    all_files_test = all_files[400:500]

    ####Shuffle the order by file name. The train has a list containing 80% of the data count. The test has 20%.
    # Step 2: Split files into train and test sets
    seed_value = 42  # Replace with actual seed value if needed
    np.random.seed(seed_value)
    # train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)

    # Process training files
    data_train = []
    labels = []
    process_files_test(all_files_train, data_train, labels, directory, points_per_sample, overlap)

    data_test = []
    labels = []
    process_files_test(all_files_test, data_test, labels, directory, points_per_sample, overlap)
    # Process testing files
    # test_data = []
    # test_labels = []
    # process_open_files(test_files, test_data, test_labels, directory, points_per_sample, overlap,
    #                    num_noises_per_sample=0)

    # Convert lists to numpy arrays and reshape appropriately
    final_data_train = np.array(data_train).reshape(-1, 2, 5000)
    final_data_test = np.array(data_test).reshape(-1, 2, 5000)  # Reshape to (num_files*num_samples) x 2 x 5000
    final_labels = np.array(labels)

    # final_test_data = np.array(test_data).reshape(-1, 2, 5000)  # Reshape to (num_files*num_samples) x 2 x 5000
    # final_test_labels = np.array(test_labels)
    return final_data_train, final_data_test


def get_dataloader_test(signal_repre, directory_path):
    print('#----------Preparing for the dataset----------#')
    X_train, X_val = read_and_process_signals_test(directory_path)
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train_ul, Y_train_ul, test_size=0.2, random_state=30)
    data = EasyDict()
    if signal_repre == 'origin':
        X_train = norm_data(X_train)
        train_dataset = TensorDataset(torch.Tensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val), )
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'dwt':
        X_train_dwt = gendwt(X_train)
        X_val_dwt = gendwt(X_val)
        X_train_dwt, X_val_dwt = norm_data(X_train_dwt), norm_data(X_val_dwt)
        train_dataset = TensorDataset(torch.Tensor(X_train_dwt))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val_dwt))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=True)
        data.train = train_loader
        data.test = val_loader
    elif signal_repre == 'fft':
        X_train_fft = genfft(X_train)
        X_val_fft = genfft(X_val)
        X_train_fft, train_norm_par = norm_data(X_train_fft)
        X_val_fft, val_norm_par = norm_data(X_val_fft)
        train_dataset = TensorDataset(torch.Tensor(X_train_fft), torch.Tensor(train_norm_par))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_fft), torch.Tensor(val_norm_par))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader


    elif signal_repre == 'time_mask':
        X_train_tm = time_mask(X_train)
        X_val_tm = time_mask(X_val)
        train_dataset = TensorDataset(torch.Tensor(X_train_tm))
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  num_workers=0, shuffle=False)

        val_dataset = TensorDataset(torch.Tensor(X_val_tm))
        val_loader = DataLoader(val_dataset, batch_size=16,
                                num_workers=0, shuffle=False)
        data.train = train_loader
        data.test = val_loader

    return data


if __name__ == "__main__":
    directory_path = "C:/Users/zhiwei/Desktop/xf/signal/part2/1/group2/train_data"
    train_data, train_labels, test_data, test_labels, test_files = read_and_process_signals(directory_path,
                                                                                            time_shift_flag=True)
    print("Data matrix shape:", train_data.shape)
    print("Labels shape:", train_labels.shape)
    signal = np.random.rand(2, 1000)  # 10 个samples, each sample has 2 channels, and each channel has 1000 data points

    shifted_signal = random_shift(signal)
    print(shifted_signal.shape)  # Output the shape of the translated signal
