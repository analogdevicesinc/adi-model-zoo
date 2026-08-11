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

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from ai_edge_litert.interpreter import Interpreter
from data_processing.data_processor import preprocess

def _get_raw_data(input_path, time=False):
    """
    Read CSV (with ';' as separator) and return its data as a numpy array.

    Args:
        input_path (str): Path to the CSV file with time and amplitude columns.
        time (bool): If True, return both time and amplitude columns; otherwise
            return amplitude only.

    Returns:
        tuple: (raw_data, sr) where raw_data is a numpy array of the selected
            columns and sr is the estimated sampling rate in Hz.
    """
    df_raw = pd.read_csv(input_path, sep=';', header=None)
    df_raw.rename(
        columns={0: 'Time', 1: 'Amplitude'},
        inplace=True
        )

    if time:
        raw_data = df_raw[['Time', 'Amplitude']]
    else:
        raw_data = df_raw['Amplitude']

    raw_data = raw_data.to_numpy()
    sr = int(df_raw.shape[0] / (df_raw.iloc[-1]['Time'] + df_raw.iloc[1]['Time']))

    return raw_data, sr

def _get_windows(df_raw, n_fft=1024, hop_length=512, frames=32):
    """
    Compute sliding window spans and their midpoints over a 1-D signal.

    Window parameters must match those used during preprocessing so that each
    window aligns with the corresponding model output. Defaults mirror the
    preprocessing defaults in data_processor.preprocess.

    Args:
        df_raw (np.ndarray): 1-D amplitude array.
        n_fft (int): Window size in samples. Must match the preprocessing n_fft.
        hop_length (int): Step size between consecutive windows in samples. Must
            match the preprocessing hop_length.
        frames (int): Number of hop-length frames grouped into each output window.
            Must match the preprocessing frames.

    Returns:
        tuple: (array_len, windows, midpoints) where array_len is the total
            number of samples, windows is a list of (start, end) index pairs,
            and midpoints is a list of center indices for each window.
    """
    array_len = len(df_raw)
    window_size = n_fft
    slide_amount = hop_length

    num_of_windows = math.floor(array_len/slide_amount) + 1

    if num_of_windows == 1:
        windows = [(0, array_len)]
        midpoints = [(array_len-1)//2]
        return array_len, windows, midpoints

    start_indices = [
        int(slide_amount * i) for i in range(num_of_windows)
    ]
    end_indices = [
        int(start + window_size) for start in start_indices
    ]
    windows = list(zip(start_indices, end_indices))

    actual_num_windows = (num_of_windows-64) // frames+1
    if actual_num_windows < 1:
        windows = [(0, array_len)]
    else:
        windows = [(windows[i*frames][0], windows[(i*frames)+(64-1)][-1]) for i in range(actual_num_windows)]
        windows[-1] = windows[-1][0], array_len # Clip end index to end of array
    
    midpoints = [(s + e) // 2 for s, e in windows]
    return array_len, windows, midpoints

def tflite_forward(interpreter, input_data):
    """
    Run forward inference on a TFLite interpreter and return all outputs.

    Handles int8 quantization and dequantization automatically based on
    the tensor metadata.

    Args:
        interpreter (Interpreter): An allocated TFLite interpreter instance.
        input_data (list[np.ndarray]): List of input arrays, one per model input.

    Returns:
        list[np.ndarray]: List of output arrays, one per model output, dequantized
            to float32 when the output type is int8.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for index, data in enumerate(input_data):
        if input_details[index]["dtype"] == np.int8:
            input_scale, input_zero_point = input_details[index]["quantization"]
            data = data / input_scale + input_zero_point
            data = np.clip(data, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[index]['index'], data)

    interpreter.invoke()

    output = []
    for index in range(len(output_details)):
        out_data = interpreter.get_tensor(output_details[index]['index'])
        if output_details[index]["dtype"] == np.int8:
            output_scale, output_zero_point = output_details[index]["quantization"]
            out_data = output_scale * (out_data.astype(np.float32) - output_zero_point)
        output.append(out_data)

    return output

def generate_loss(tflite_model, input_data):
    """
    Compute per-sample anomaly scores via negative log likelihood.

    Runs forward inference on each sample and applies log-softmax over the
    model logits. Returns the negative log probability of class 0
    (machine_id_00) as the anomaly score for each sample.

    Args:
        tflite_model (str): Path to the TFLite model file.
        input_data (np.ndarray): Preprocessed input array of shape (N, ...) where
            N is the number of samples.

    Returns:
        np.ndarray: 1-D array of shape (N,) containing the negative log likelihood
            for each sample. Higher values indicate greater anomaly likelihood.
    """
    interpreter = Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    num_samples = input_data.shape[0]
    model_outputs = np.zeros((input_data.shape[0], 4))

    for i in range(num_samples):

        model_output = tflite_forward(interpreter, np.expand_dims(input_data[i], axis=0))
        model_outputs[i, :] = model_output[0]

    logits = model_outputs

    # Log-softmax
    max_logit = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logit
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp

    # Negative log probability
    losses = -log_probs[:, 0]
    np.expand_dims(losses, axis=0)
    
    return losses

def compute_anomalies(input_csv, losses, threshold):
    """
    Map per-window anomaly losses back to per-sample annotations.

    Reads the raw CSV signal, splits it into windows, and marks each sample
    as anomalous if its window's loss exceeds the given threshold.

    Args:
        input_csv (str): Path to the input CSV file.
        losses (np.ndarray): 1-D array of per-window negative log likelihood scores.
        threshold (float): Loss value above which a window is considered anomalous.

    Returns:
        tuple: (raw_data, anomaly_spans, midpoints, sr, df_summary) where
            raw_data is the time-amplitude array, anomaly_spans lists (start, end)
            index pairs for anomalous windows, midpoints are window center indices,
            sr is the sampling rate, and df_summary is a DataFrame with per-sample
            time, amplitude, loss, and anomaly flag columns.
    """
    # Extract raw data and windows
    raw_data, sr = _get_raw_data(input_csv, time=True)
    array_len, windows, midpoints = _get_windows(raw_data[:, 1])

    # Isolate windows that are anomalous (NLL above threshold)
    anomaly_spans = [windows[i] for i in range(len(windows)) if (losses[i] > threshold)]

    loss_per_sample = np.zeros(array_len)
    is_anomaly = np.zeros(array_len)

    for i, (s, e) in enumerate(windows):
        s = max(0, s)
        e = min(array_len, e)

        # Tag NLL per sample. Use max in case of overlapping windows
        loss_per_sample[s:e] = np.maximum(loss_per_sample[s:e], losses[i])
    
    for i, (s, e) in enumerate(anomaly_spans):
        s = max(0, s)
        e = min(array_len, e)

        # Tag anomalous samples based on anomaly spans
        is_anomaly[s:e] = 1

    # Create dataframe to summarize output
    df_summary = pd.DataFrame(
        {
            "time": raw_data[:, 0],
            "amplitude": raw_data[:, 1],
            "negative_log_likelihood": loss_per_sample,
            "is_anomaly": is_anomaly
        }
    )

    return raw_data, anomaly_spans, midpoints, sr, df_summary

def display_loss(midpoints, sr, losses, output_path):
    """
    Plot the negative log likelihood over time and save to a JPEG file.

    Args:
        midpoints (list[int]): Center sample indices for each window.
        sr (int): Sampling rate in Hz, used to convert indices to seconds.
        losses (np.ndarray): 1-D array of per-window NLL scores.
        output_path (str): Output file path without extension (.jpg is appended).
    """
    # Plot NLL data
    plt.figure(figsize=(12,6))
    plt.title("Anomaly Detection - Negative Log Likelihood")
    plt.plot((np.asarray(midpoints)/sr), losses, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Negative Log Likelihood')

    plt.savefig(output_path + ".jpg")

def display_anomalies(raw_data, anomaly_spans, threshold, output_path):
    """
    Plot the raw signal with red overlays on anomalous regions and save to JPEG.

    Args:
        raw_data (np.ndarray): 2-D array of shape (N, 2) with time and amplitude.
        anomaly_spans (list[tuple[int, int]]): (start, end) index pairs for
            anomalous windows.
        threshold (float): NLL threshold used for detection, shown in the legend.
        output_path (str): Output file path without extension (.jpg is appended).
    """
    # Plot input data
    plt.figure(figsize=(12,6))
    plt.title("Anomaly Detection")
    plt.plot(raw_data[:, 0], raw_data[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Normalized)')

    # Add red overlay over areas with loss at given threshold
    red = patches.Patch(color='red', alpha=0.5, label=f'fault detected at NLL threshold: {threshold}')
    plt.legend(handles=[red])

    for anomaly in anomaly_spans:
        s, e = anomaly
        plt.axvspan(raw_data[s, 0], raw_data[(e-1), 0], color='red', alpha=0.1)

    plt.savefig(output_path + ".jpg")

def save_output(summary, output_path):
    """
    Save the anomaly detection summary DataFrame to a CSV file.

    Args:
        summary (pd.DataFrame): Per-sample summary with time, amplitude, loss,
            and anomaly flag columns.
        output_path (str): Output file path without extension (.csv is appended).
    """
    # Save output summary dataframe to CSV file
    summary.to_csv(output_path+".csv")

def detect_anomalies(tflite_model, csv_path, output_folder):
    """
    Run the full anomaly detection pipeline on a single CSV input.

    Preprocesses the signal, computes per-window NLL scores, identifies
    anomalous regions, and saves loss/anomaly plots and a summary CSV.

    Args:
        tflite_model (str): Path to the TFLite model file.
        csv_path (str): Path to the input CSV file with time-amplitude data.
        output_folder (list[str]): Two-element list of output path stems:
            [loss_output, anomaly_output] (extensions are appended automatically).
    """
    # Set the threshold for anomaly detection here
    THRESHOLD = 0.5

    # Output path for loss graph and anomaly detection graph
    loss_output, anomaly_output = output_folder
    
    input_data, _ = preprocess(csv_path)
    loss = generate_loss(tflite_model,
                         input_data)
    
    raw_data, anomaly_spans, midpoints, sr, summary = compute_anomalies(
            input_csv=csv_path,
            losses=loss,
            threshold=THRESHOLD,
        )

    display_anomalies(raw_data=raw_data,
                      anomaly_spans=anomaly_spans,
                      threshold=THRESHOLD,
                      output_path=anomaly_output)
    
    display_loss(midpoints=midpoints,
                 sr=sr,
                 losses=loss,
                 output_path=loss_output)
    
    save_output(summary=summary,
                output_path=anomaly_output)
    
    print(f"Computation of NLL complete. Output saved to {loss_output}.")
    print(f"Anomaly detection complete. Output saved to {anomaly_output}.")

def main(argv):
    """
    Basic Micronet (Anomaly Detection) TFLite Example
    Usage: python example.py <csv file> [<csv file> ...]
    """

    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    tflite_path   = os.path.join(root_path, "data", "model", "micronet_ad_small_int8.tflite")
    csvs = argv or [os.path.join(root_path, "data", "input", "anomalous_data.csv")]

    for csv_file in csvs:

        base_name = Path(csv_file).stem
        loss_output_name = os.path.join(output_folder, "nll_" + base_name)
        anomaly_output_name = os.path.join(output_folder, "detected_" + base_name)
        output_folder = [loss_output_name, anomaly_output_name]
        detect_anomalies(tflite_path, csv_file, output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])
