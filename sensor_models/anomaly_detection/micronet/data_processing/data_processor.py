from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft as scipy_stft
from scipy.ndimage import zoom
import scipy
import argparse
import os
import sys

def _quantize(x, scale, zero_point):
    return (np.round(x / scale) + zero_point).astype(np.int8)

def _load_csv(fname: os.PathLike | str):
    df_raw = pd.read_csv(fname, sep=';', header=None)
    df_raw.rename(
        columns={0: 'Time', 1: 'Amplitude'},
        inplace=True
        )

    raw_data = df_raw['Amplitude']
    raw_data = raw_data.to_numpy()
    sr = int(df_raw.shape[0] / (df_raw.iloc[-1]['Time'] + df_raw.iloc[1]['Time']))
    return raw_data.astype(np.float32), sr

def _mel_filter_bank(n_mels, n_fft_bins, sample_rate, lower_hz, upper_hz):
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(lower_hz), hz_to_mel(upper_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft_bins * 2 - 1 + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_fft_bins, n_mels), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
        for j in range(left, center):
            if center != left:
                filters[j, i] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filters[j, i] = (right - j) / (right - center)
    return filters

def _compute_log_mel(audio, n_fft, hop_length, n_mels, sample_rate):
    _, _, Zxx = scipy_stft(audio, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx.T)

    n_fft_bins = magnitude.shape[-1]
    mel_weights = _mel_filter_bank(n_mels, n_fft_bins, sample_rate, 0.0, sample_rate / 2.0)

    mel = magnitude @ mel_weights
    log_mel = np.log(mel + 1e-6)

    return log_mel

def preprocess(fname: os.PathLike | str, **kwargs):
    """
    Preprocesses time-series CSV file into a (n_windows, 1, 32, 32, 1) spectrogram.

    Args:
        fname (Path | str): File name of the CSV file.

    Returns:
        np.ndarray with shape (n_windows, 1, 32, 32, 1) and type int8 to be streamed to the MCU.
    """

    # Initialize variables
    target_sampling_rate_Hz = kwargs.get('target_sampling_rate_Hz', 16000)
    n_mels = kwargs.get('n_mels', 64)
    frames = kwargs.get('frames', 32)
    n_fft = kwargs.get('n_fft', 1024)
    hop_length = kwargs.get('hop_length', 512)
    scale = kwargs.get('scale', 0.02277415059506893)
    zero_point = kwargs.get('zero_point', 3)

    if os.path.splitext(fname)[-1] == ".csv":
        raw_data, sr = _load_csv(fname)
    else:
        raise ValueError("Unsupported file extension.")
    
    # Resample signal to desired sampling rate
    if sr != target_sampling_rate_Hz:
        raw_data = scipy.signal.resample(raw_data, target_sampling_rate_Hz)

    if len(raw_data) < target_sampling_rate_Hz:
        padding = target_sampling_rate_Hz - len(raw_data)
        raw_data = np.pad(raw_data, (0, padding))
    
    # Generate log mel spectrogram
    log_mel_spectrogram = _compute_log_mel(audio=raw_data,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           n_mels=n_mels,
                                           sample_rate=sr)

    # Pad spectrogram if less than segment frames (64)
    T = log_mel_spectrogram.shape[0]
    segment_frames = 64
    if T <= segment_frames:
        padded = np.zeros((segment_frames, n_mels), dtype=np.float32)
        padded[:T] = log_mel_spectrogram
        log_mel_spectrogram = padded
        T = segment_frames

    # Window the spectrogram into (64, 64) chunks
    segments = []
    for start in range(0, T - segment_frames + 1, frames):
        seg = log_mel_spectrogram[start:start + segment_frames]

        # Resize each chunk into (32, 32)
        zoom_factors = (32 / seg.shape[0], 32 / seg.shape[1])
        resized = zoom(seg, zoom_factors, order=1)[..., np.newaxis]
        segments.append(resized)

    segments = np.array(segments, dtype=np.float32)
    segments = np.expand_dims(segments, axis=1)

    output = _quantize(segments, scale, zero_point)
    ref_output = np.reshape(output, (output.shape[0], output.shape[2]*output.shape[3]))

    return output, ref_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the input csv file")
    parser.add_argument("--output_file_path", required=True, help="absolute file path to save the output .npy file")
    parser.add_argument("--inference_file_path", required=True, help="absolute file path to save the output  xref .npy file")
    parser.add_argument("--quant_type",required=True, help="Quantization parameter (int8,int16,float32)")
    args = parser.parse_args()
    source_path = Path(args.source)
    output_file = Path(args.output_file_path)
    inference_file = Path(args.inference_file_path)

    x, x_ref = preprocess(source_path)
    print(x_ref.shape)
    np.save(output_file, x)
    np.save(inference_file, x_ref)

    print(f"Preprocessed bitstream shape: {x.shape}. Batch size: {x.shape[0]}. Stream length in bytes of four: {x.shape[1]}.")
    print(f"Saved to: {output_file}")
