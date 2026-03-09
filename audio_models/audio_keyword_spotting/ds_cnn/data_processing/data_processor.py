# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
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
# ==============================================================================
#
# Modified to use TensorFlow 2.0 and data pipelines.
#/* Portions Copyright (c) 2026 Analog Devices, Inc. */

import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
import numpy as np
import librosa

INPUT_QUANTIZATION_PARAMETERS = {
    'float32': {'scale': 1.0, 'zero_point': 0},
    'int16': {'scale': 0.007542324718087912, 'zero_point': 0},
    'int8': {'scale': 1.084193468093872, 'zero_point': 100},
}

OUTPUT_QUANTIZATION_PARAMETERS = {
    'float32': {'scale': 1.0, 'zero_point': 0},
    'int16': {'scale': 3.0517578125e-05, 'zero_point': 0},
    'int8': {'scale': 0.00390625, 'zero_point': -128},
}

def _quantize(input_data, dtype):
    if dtype not in INPUT_QUANTIZATION_PARAMETERS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    params = INPUT_QUANTIZATION_PARAMETERS[dtype]
    scale = params['scale']
    zero_point = params['zero_point']
    
    quantized = input_data / scale + zero_point
    
    if dtype == 'int16':
        quantized = np.round(quantized).astype(np.int16)
    elif dtype == 'int8':
        quantized = np.round(quantized).astype(np.int8)
    else:
        quantized = np.array(quantized, dtype=np.float32)
    
    return quantized

def _dequantize(input_data, dtype):
    if dtype not in OUTPUT_QUANTIZATION_PARAMETERS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    params = OUTPUT_QUANTIZATION_PARAMETERS[dtype]
    scale = params['scale']
    zero_point = params['zero_point']
    
    dequantized = (input_data - zero_point) * scale
    
    return dequantized.astype(np.float32)

def _calculate_mfcc(audio_signal, audio_sample_rate, window_size, window_stride, num_mfcc):
    """Returns Mel Frequency Cepstral Coefficients (MFCC) for a given audio signal.

    Args:
        audio_signal: Raw audio signal in range [-1, 1]
        audio_sample_rate: Audio signal sample rate
        window_size: Window size in samples for calculating spectrogram
        window_stride: Window stride in samples for calculating spectrogram
        num_mfcc: The number of MFCC features wanted.

    Returns:
        Calculated mffc features.
    """
    spectrogram = audio_ops.audio_spectrogram(input=audio_signal, window_size=window_size, stride=window_stride,
                                              magnitude_squared=True)

    mfcc_features = audio_ops.mfcc(spectrogram, audio_sample_rate, dct_coefficient_count=num_mfcc)

    return mfcc_features

def preprocess(fname, **kwargs):
    """Preprocess audio files for keyword spotting. Converts audio to MFCC features.
    
    Args:
        fname: Audio file (*.wav) to be preprocessed.
        samples: Number of samples to load from audio file.
        window_size: Window size in samples for calculating spectrogram.
        window_stride: Window stride in samples for calculating spectrogram.
        num_mfcc: Number of MFCC features to calculate.

    Returns:
        If save_dir is provided, returns the save directory path.
        If only one audio file is processed, returns the preprocessed data as a numpy array.
        Otherwise, returns a list of preprocessed data arrays.
    """

    if not fname.endswith('.mp3') and not fname.endswith('.wav'):
        raise ValueError("Unsupported file format. Please provide a .mp3 or .wav file.")

    # Main preprocessing
    samples = kwargs.get('samples', 16000)
    window_size = kwargs.get('window_size', 640)
    window_stride = kwargs.get('window_stride', 320)
    num_mfcc = kwargs.get('num_mfcc', 10)
    
    audio_signal, sample_rate = librosa.load(fname, sr=samples)

    if len(audio_signal) < samples:
        audio_signal = np.pad(audio_signal, (0, samples - len(audio_signal)), mode='constant')
    else:
        audio_signal = audio_signal[:samples]
        
    audio_signal = tf.expand_dims(audio_signal, axis=1)
    mfcc_features = _calculate_mfcc(audio_signal=audio_signal,
                                    audio_sample_rate=sample_rate,
                                    window_size=window_size,
                                    window_stride=window_stride,
                                    num_mfcc=num_mfcc)

    mfcc_flat = tf.reshape(mfcc_features, [1, -1]) # Flatten

    return _quantize(mfcc_flat.numpy(), dtype=kwargs.get('dtype', 'float32'))

def postprocess(fname, **kwargs):
    try:
        y = np.load(fname).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error loading file {fname}: {e}")

    return _dequantize(y, dtype=kwargs.get('dtype', 'float32'))

if __name__ == "__main__":
    SAMPLE_INPUT = "./sample_data/stop.wav"
    SAMPLE_OUTPUT = "./sample_data/output_int8.npy"

    x = preprocess(SAMPLE_INPUT, dtype='int8')
    print(f"Mel spectrograms created with shape {x.shape} and type {x.dtype}")

    y = postprocess(SAMPLE_OUTPUT, dtype='int8')
    print(f"Postprocessed output with shape {y.shape} and type {y.dtype}")
    print(f"Output values: {y}")
    print(f"Output sum: {np.sum(y)}")
