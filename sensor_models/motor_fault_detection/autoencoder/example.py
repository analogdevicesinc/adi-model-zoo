# Copyright © 2025 Analog Devices, Inc. All Rights Reserved. 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from pathlib import Path

import numpy as np
from numpy.fft import fft
from torch import nn
from pathlib import Path
import pandas as pd
import scipy
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference

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

def _get_raw_data(input_path, time=False):
        
    df_raw = pd.read_csv(input_path, sep=';', header=None)
    df_raw.rename(
        columns={0: 'Time', 1: 'Voltage_x', 2: 'Voltage_y',
                    3: 'Voltage_z', 4: 'x', 5: 'y', 6: 'z'},
        inplace=True
        )
    
    ss_vibr_x1 = 0.0
    ss_vibr_y1 = 0.0
    ss_vibr_z1 = 0.0
    try:
        ss_vibr_x1 = df_raw.iloc[0]['x']
        ss_vibr_y1 = df_raw.iloc[0]['y']
        ss_vibr_z1 = df_raw.iloc[0]['z']

    except KeyError:
        pass

    df_raw["Acceleration_x (g)"] = 50 * (df_raw["Voltage_x"] - ss_vibr_x1)
    df_raw["Acceleration_y (g)"] = 50 * (df_raw["Voltage_y"] - ss_vibr_y1)
    df_raw["Acceleration_z (g)"] = 50 * (df_raw["Voltage_z"] - ss_vibr_z1)

    if time:
        raw_data = df_raw[["Time", "Acceleration_x (g)", "Acceleration_y (g)", "Acceleration_z (g)"]]
    else:
        raw_data = df_raw[["Acceleration_x (g)", "Acceleration_y (g)", "Acceleration_z (g)"]]

    return raw_data.to_numpy()

def preprocess(input_path):

    # Initialize variables
    sensor_sr_Hz = 20000
    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75
    target_sampling_rate_Hz = 2000
    cnn_1dinput_len = 256
    num_start_zeros = 3
    num_end_zeros = 10

    raw_data = _get_raw_data(input_path)

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

    file_cnn_signals = file_cnn_signals.transpose([0, 2, 1])
    reference = _quantize(file_cnn_signals, 255, 128)

    return reference.astype(np.float32)  

def generate_loss(reconstructed, reference):

    window_size = reconstructed.shape[0]
    losses = np.zeros((1, window_size))

    for i in range(window_size):
        losses[0, i] = _get_mse_loss(reconstructed[i], reference[i])

    return losses

def reconstruct_signal(ai8x_model, input_data):

    model = AI8XInference(model_class="ai85autoencoder",
                    model_module="ai85net-autoencoder",
                    model_checkpoint_path=Path(ai8x_model),
                    module=True)
    
    output = np.zeros(shape=input_data.shape, dtype=input_data.dtype)
    for i in range(len(input_data)):
        output_chunk = model.generate_expected_output(input_data[i])
        output_chunk = output_chunk.detach().numpy()
        output[i] = output_chunk
    
    return output

def display_reconstruction(reconstructed, reference,  output_path, window_idx=0, channel_idx=0):
    
    fs = 2000.0
    N  = 500
    K  = reconstructed.shape[1]
    freqs = np.arange(K) * (fs / N)

    # Plot reconstructed signal and reference signal on the same axis
    plt.figure(figsize=(12,6))
    plt.title("Motor Signal Reconstruction")
    plt.plot(freqs, reconstructed[window_idx,:,channel_idx])
    plt.plot(freqs, reference[window_idx,:,channel_idx])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Voltage (normalized, quantized)')
    plt.legend(['Reconstructed', 'Reference'])
    
    plt.savefig(output_path + ".jpg")

def display_faults(input_csv, losses, threshold, output_path, window_size = 500, overlap_ratio=0.75, downsampling_ratio=10.0):

    raw_data = _get_raw_data(input_csv, time=True)

    # Get indices of the windows given the raw_data length, the window_size, the overlap_ratio, and the downsampling_ratio
    array_len = raw_data.shape[0]
    window_overlap = math.ceil(window_size * overlap_ratio)
    slide_amount = window_size - window_overlap
    num_of_windows = math.floor(((array_len // downsampling_ratio) - window_size) / slide_amount) + 1
    start_indices = [int(slide_amount * i * downsampling_ratio) for i in range(num_of_windows)]
    end_indices = [int(start + window_size * downsampling_ratio) for start in start_indices]
    
    # List of windows (start_idx, end_idx)
    windows = list(zip(start_indices, end_indices))
        
    plt.figure(figsize=(12,6))
    plt.title("Motor Fault Detection")
    plt.plot(raw_data[:, 0], raw_data[:, 1]) # Plot acceleration x-axis only
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    red = patches.Patch(color='red', alpha=0.5, label=f'fault detected at reconstruction loss threshold: {threshold}')
    plt.legend(handles=[red])
    
    # Get anomaly spans
    anomaly_spans = []

    for i in range(losses.shape[-1]):
        if losses[0, i] > threshold:
            anomaly_spans.append(windows[i])
    
    # Merge overlapping spans
    spans = sorted(anomaly_spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        _, last_e = merged[-1]
        if s <= last_e:  # overlap or touching
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    anomaly_spans = [tuple(x) for x in merged]

    for s, e in anomaly_spans:
        plt.axvspan(raw_data[s, 0], raw_data[(e-1), 0], color='red', alpha=0.5)

    plt.savefig(output_path + ".jpg")

def save_output(reconstructed, loss, reconstruct_output, fault_output):

    np.save(reconstruct_output + ".npy", reconstructed)
    np.save(fault_output + ".npy", loss)
        
def detect_faults(ai8x_model, input_path, output_folder):

    reconstruct_output, fault_output = output_folder
    
    input_data = preprocess(input_path)
    reconstructed_signal = reconstruct_signal(ai8x_model, input_data)
    loss = generate_loss(reconstructed=reconstructed_signal,
                         reference=input_data)

    display_reconstruction(reconstructed=reconstructed_signal,
                           reference=input_data,
                           output_path=reconstruct_output)
    
    display_faults(input_csv=input_path,
                   losses=loss,
                   threshold=0.115,
                   output_path=fault_output)
    
    save_output(reconstructed=reconstructed_signal,
                loss=loss,
                reconstruct_output=reconstruct_output,
                fault_output=fault_output)
    
    print(f"Reconstruction complete. Output saved to {reconstruct_output}")
    print(f"Fault detection complete. Output saved to {fault_output}")
    
def main(argv):
    """
    Basic Autoencoder ai8x Example
    Usage: python example.py <csv file> [<csv file> ...]
    """

    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    ai8x_path = os.path.join(root_path, "data", "model", "ai85-autoencoder-samplemotordatalimerick-qat-q.pth.tar")
    csvs = argv or [os.path.join(root_path, "data", "input", "motor.csv")]

    for csv_file in csvs:

        base_name = Path(csv_file).stem
        reconstruction_output_name = os.path.join(output_folder, "reconstructed_" + base_name)
        fault_output_name = os.path.join(output_folder, "detected_" + base_name)
        output_folder = [reconstruction_output_name, fault_output_name]
        detect_faults(ai8x_path, csv_file, output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])
