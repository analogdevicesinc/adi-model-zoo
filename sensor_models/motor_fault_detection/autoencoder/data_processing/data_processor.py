from numpy.fft import fft
from torch import nn
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
import os
import math
import shutil
import yaml

def _sliding_windows_1d_get_indices(array_len, window_size, overlap_ratio, downsampling_ratio=10):
    """
    Get start and end indices of one dimensional array windowed
    in window_size length according to overlap ratio.
    """

    window_overlap = math.ceil(window_size * overlap_ratio)

    slide_amount = window_size - window_overlap
    num_of_windows = math.floor(((array_len // downsampling_ratio) - window_size) / slide_amount) + 1

    start_indices = [int(slide_amount * i * downsampling_ratio) for i in range(num_of_windows)]
    end_indices = [int(start + window_size * downsampling_ratio) for start in start_indices]

    return list(zip(start_indices, end_indices))

def _sliding_windows_1d(array, window_size, overlap_ratio):
    """
    One dimensional array is windowed and returned
    in window_size length according to overlap ratio.
    """

    window_overlap = math.ceil(window_size * overlap_ratio)

    slide_amount = window_size - window_overlap
    num_of_windows = math.floor((len(array) - window_size) / slide_amount) + 1

    result_list = np.zeros((num_of_windows, window_size))

    for i in range(num_of_windows):
        start_idx = slide_amount * i
        end_idx = start_idx + window_size
        result_list[i] = array[start_idx:end_idx]

    return result_list

def _sliding_windows_on_columns_of_2d(array, window_size, overlap_ratio):
    """
    Two dimensional array is windowed and returned
    in window_size length according to overlap ratio.
    """

    array_len, num_of_cols = array.shape

    window_overlap = math.ceil(window_size * overlap_ratio)
    slide_amount = window_size - window_overlap
    num_of_windows = math.floor((array_len - window_size) / slide_amount) + 1

    result_list = np.zeros((num_of_cols, num_of_windows, window_size))

    for i in range(num_of_cols):
        result_list[i, :, :] = _sliding_windows_1d(
            array[:, i],
            window_size, overlap_ratio
            )

    return result_list

def _split_file_raw_data(file_raw_data, file_raw_data_fs_in_Hz, duration_in_sec, overlap_ratio):
    """
    Raw data is split into windowed data.
    """

    num_of_samples_per_window = int(file_raw_data_fs_in_Hz * duration_in_sec)

    sliding_windows = _sliding_windows_on_columns_of_2d(
        file_raw_data,
        num_of_samples_per_window,
        overlap_ratio
        )

    return sliding_windows

def _normalize_signal(features):
    """
    Normalize signal with Local Min Max Normalization
    """
    # Normalize data:
    for instance in range(features.shape[0]):
        instance_max = np.max(features[instance, :, :], axis=1)
        instance_min = np.min(features[instance, :, :], axis=1)

        for feature in range(features.shape[1]):
            for signal in range(features.shape[2]):
                features[instance, feature, signal] = (
                    (features[instance, feature, signal] - instance_min[feature]) /
                    (instance_max[feature] - instance_min[feature])
                )

    return features

def _quantize(x, scale, zero_point):
    return (x * scale - zero_point).astype(np.int8)
    

def _pack_bitstream(istream):
    """
    Pack input array into bitstream for max78002
    """
    SCALE = 255
    ZEROPOINT = 128

    input = _quantize(istream, SCALE, ZEROPOINT)

    if input.shape != (256, 3):
        raise ValueError("Input shape is not correct")
    
    x = input.reshape(4, 64, 3)
    x = np.transpose(x, (2,1,0))
    x = x.reshape(3, 16, 4, 4)
    x = np.transpose(x, (1,0,3,2))
    x = x.reshape(16, 12, 4)
    x = np.flip(x, 2)
    x = x.reshape(192,4)

    merged = np.zeros(192).astype(np.uint32)

    for i in range(192):
        merged[i] = ((x[i][0] & 0xFF) << 24) | \
                    ((x[i][1] & 0xFF) << 16 ) | \
                    ((x[i][2] & 0xFF) << 8 ) | \
                    ((x[i][3] & 0xFF))

    return merged

def _unpack_bitstream(istream):
    # Extract values from bitstream
    reconstructed = np.zeros(768).astype(np.int8)
    j = 0
    for y in istream:
        reconstructed[j+1] = ((y >> 16) & 0xFFFF)
        reconstructed[j] = y & 0xFFFF
        j += 2

    return reconstructed.reshape(256,3)

def _get_mse_loss(input, target):
    """
    Compute MSE loss of two quantized (256, 3) FFTs
    """
    input = input / 2**7
    target = target / 2**7

    y1 = torch.from_numpy(input)
    y0 = torch.from_numpy(target)

    y1 = torch.reshape(y1, (1, 256, 3))
    y0 = torch.reshape(y0, (1, 256, 3))

    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(y1, y0)
    loss_numpy = loss.cpu().detach().numpy()
    loss_numpy.shape
    decay_vector = np.array([1**i for i in range(loss_numpy.shape[2])])
    decay_vector = np.tile(decay_vector, (loss_numpy.shape[0], loss_numpy.shape[1], 1))
    decayed_loss = loss_numpy * decay_vector

    return decayed_loss.mean(axis=(1, 2))[0]

def preprocess(fname: os.PathLike | str, **kwargs):
    """
    Preprocesses time-series CSV file into n windows of (256, 3) FFT window.

    Args:
        fname (Path | str): File name of the CSV file.

    Returns:
        np.ndarray with shape (n, 192) the n windows of 192 uint32 to be streamed to the MCU.
    """

    # Initialize variables
    sensor_sr_Hz = kwargs.get('sensor_sr_Hz', 20000)
    signal_duration_in_sec = kwargs.get('signal_duration_in_sec', 0.25)
    overlap_ratio = kwargs.get('overlap_ratio', 0.75)
    target_sampling_rate_Hz = kwargs.get('target_sampling_rate_Hz', 2000)
    cnn_1dinput_len = kwargs.get('cnn_1dinput_len', 256)
    num_start_zeros = 3
    num_end_zeros = 10

    accepted_file_ext = [".csv"]

    if os.path.splitext(fname)[-1] not in accepted_file_ext:
        raise ValueError("Unsupported file extension.")
        
    df_raw = pd.read_csv(fname, sep=';', header=None)
    df_raw.rename(
        columns={0: 'Time', 1: 'Voltage_x', 2: 'Voltage_y',
                    3: 'Voltage_z', 4: 'x', 5: 'y', 6: 'z'},
        inplace=True
        )
    
    ss_vibr_x1 = 0.0
    ss_vibr_y1 = 0.0
    ss_vibr_z1 = 0.0
    try:
        ss_vibr_x1 = df_raw.iloc[0]['x'] if not np.isnan(df_raw.iloc[0]['x']) else 0.0
        ss_vibr_y1 = df_raw.iloc[0]['y'] if not np.isnan(df_raw.iloc[0]['y']) else 0.0
        ss_vibr_z1 = df_raw.iloc[0]['z'] if not np.isnan(df_raw.iloc[0]['z']) else 0.0

    except KeyError:
        pass

    df_raw["Acceleration_x (g)"] = 50 * (df_raw["Voltage_x"] - ss_vibr_x1)
    df_raw["Acceleration_y (g)"] = 50 * (df_raw["Voltage_y"] - ss_vibr_y1)
    df_raw["Acceleration_z (g)"] = 50 * (df_raw["Voltage_z"] - ss_vibr_z1)

    raw_data = df_raw[["Acceleration_x (g)", "Acceleration_y (g)", "Acceleration_z (g)"]]
    raw_data = raw_data.to_numpy()

    downsampling_ratio = sensor_sr_Hz / target_sampling_rate_Hz
    new_sampling_rate = int(sensor_sr_Hz/ downsampling_ratio)

    file_raw_data_sampled = scipy.signal.decimate(raw_data, int(downsampling_ratio), axis=0)

    file_raw_data_windows = _split_file_raw_data(
        file_raw_data_sampled,
        new_sampling_rate,
        signal_duration_in_sec,
        overlap_ratio
        )

    num_features = file_raw_data_windows.shape[0]
    num_windows = file_raw_data_windows.shape[1]

    fft_output_window_size = cnn_1dinput_len

    file_cnn_signals = np.zeros((num_features, num_windows, fft_output_window_size))

    for window in range(num_windows):
        for feature in range(num_features):

            signal_for_fft = file_raw_data_windows[feature, window, :]

            fft_out = abs(fft(signal_for_fft))
            fft_out = fft_out[:fft_output_window_size]

            fft_out[:num_start_zeros] = 0
            fft_out[-num_end_zeros:] = 0

            file_cnn_signals[feature, window, :] = fft_out

        file_cnn_signals[:, window, :] = file_cnn_signals[:, window, :] / \
                    np.sqrt(np.power(file_cnn_signals[:, window, :], 2).sum())
        
    file_cnn_signals = file_cnn_signals.transpose([1, 0, 2])

    file_cnn_signals = _normalize_signal(file_cnn_signals)

    file_cnn_signals[:, :, :num_start_zeros] = 0.5
    file_cnn_signals[:, :, -num_end_zeros:] = 0.5

    file_cnn_signals = file_cnn_signals.transpose([0,2,1])

    reference = _quantize(file_cnn_signals, 255, 128)

    output = np.zeros([num_windows, 192], np.uint32)
    for idx in range(num_windows):
        output[idx, :] = _pack_bitstream(file_cnn_signals[idx])

    return output, reference

def postprocess(fname: os.PathLike | str,
                reference: os.PathLike | str | np.ndarray = None,
                output_mode = 'loss',
                **kwargs):
    """
    Postprocesses reconstructed bitstream into (256, 3) quantized FFT.

    Args:
        fname (Path | str): File name of the bitstream numpy file with shape (1, 384) or (384).
        reference (Path | str | np.ndarray, optional): File name of the reference FFTs numpy file with shape (1, 384) or (384).
            If output_mode is 'loss', reference is required to compute MSE loss.

    Returns:
        np.ndarray with shape (n, 192) the n windows of 192 uint32 to be streamed to the MCU.
    """
    reconstructed_bits = np.load(fname)
    
    if len(reconstructed_bits.shape) == 2:
        window_size = reconstructed_bits.shape[0]
    else:
        window_size = 1

    if output_mode=='loss':
        if reference is None:
            raise FileNotFoundError("Reference signal in numpy format is required")
        
        if isinstance(reference, (os.PathLike, str)):
            original = np.load(reference)
        elif isinstance(reference, (np.ndarray, torch.Tensor)):
            original = reference
        else:
            raise ValueError("Reference signal instance is not supported.")
        
        losses = np.zeros((1, window_size))

        for i in range(window_size):
            reconstructed_bits_i = reconstructed_bits[i] if len(reconstructed_bits.shape) > 1 else reconstructed_bits
            # Extract values from bitstream
            reconstructed = _unpack_bitstream(reconstructed_bits_i)
            losses[0, i] = _get_mse_loss(reconstructed, original[i])

        return losses
    else:
        reconstructed = np.zeros((window_size, 256, 3)).astype(np.int8)
        # Extract values from bitstream
        for i in range(window_size):
            reconstructed_bits_i = reconstructed_bits[i] if len(reconstructed_bits.shape) > 1 else reconstructed_bits
            # Extract values from bitstream
            reconstructed[i] = _unpack_bitstream(reconstructed_bits_i)

        return reconstructed
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    SAMPLE_INPUT = "./sample_data/motor.csv"
    SAMPLE_OUTPUT = "./sample_data/sample_reconstructed_ostream.npy"

    # Preprocess input to create bitstream
    x = preprocess(SAMPLE_INPUT)
    print(f"Preprocessed bitstream shape: {x.shape}. Batch size: {x.shape[0]}. Stream length in bytes of four: {x.shape[1]}.")

    # Postprocess output to create reconstructed signal
    y = postprocess(SAMPLE_OUTPUT)
    print(f"Reconstructed signal has shape {y.shape}.")
    plt.plot(y)
    plt.show()
    plt.clf()

    # Postprocess output to calculate MSE loss from the original signal
    loss = postprocess(SAMPLE_OUTPUT, 'loss', x[20])
    print(f"MSE Loss: {loss}")
