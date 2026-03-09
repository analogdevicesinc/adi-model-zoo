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

import librosa
from librosa.feature import melspectrogram
from librosa.core import load
import numpy as np
import matplotlib.pyplot as plt

INPUT_QUANTIZATION_PARAMETERS = {
    'float32': {'scale': 1.0, 'zero_point': 0},
    'int8': {'scale': 4.748456954956055, 'zero_point': -128},
}

OUTPUT_QUANTIZATION_PARAMETERS = {
    'float32': {'scale': 1.0, 'zero_point': 0},
    'int8': {'scale': 0.00390625, 'zero_point': -128},
}

SAMPLING_RATE = 22050
GENRES  = [ 'blues',
            'classical',
            'country',
            'disco',
            'hiphop',
            'jazz',
            'metal',
            'pop',
            'reggae',
            'rock'  ]

def _quantize(input_data, dtype):
    if dtype not in INPUT_QUANTIZATION_PARAMETERS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    params = INPUT_QUANTIZATION_PARAMETERS[dtype]
    scale = params['scale']
    zero_point = params['zero_point']
    
    quantized = input_data / scale + zero_point
    
    if dtype == 'int8':
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
    
    dequantized = (input_data.astype(np.float32) - zero_point) * scale
    
    return dequantized.astype(np.float32)

def preprocess(fname, dtype='int8', **kwargs):
    """
    Preprocess input of model.
    """

    if not fname.endswith('.mp3') and not fname.endswith('.wav'):
        raise ValueError("Unsupported file format. Please provide a .mp3 or .wav file.")
    
    mel         = kwargs.get("mel", True)
    n_fft       = kwargs.get("n_fft", 2048)
    hop_length  = kwargs.get("hop_length", 512)
    
    min_size = (128 * hop_length) + n_fft - hop_length - 1536
    y, sr = load(fname, sr=SAMPLING_RATE)

    if y.size < min_size:
        y = np.pad(y, (0, min_size - y.size), mode='constant', constant_values=0)

    if mel:
        S = melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).T
        S = S[:-1 * (S.shape[0] % 128)]

        num_chunk   = S.shape[0] / 128
        data_chunks = np.split(S, num_chunk)
        data_chunks = np.array(data_chunks)
        data_chunks = data_chunks.reshape(int(num_chunk), 128, 128, 1)

    else:
        num_chunk = y.size // (min_size)
        data_chunks = np.split(y[:num_chunk*min_size], num_chunk, axis=0)
        data_chunks = np.array(data_chunks)

    return _quantize(data_chunks, dtype=dtype)

def postprocess(fname, dtype='int8', **kwargs):
    """
    Postprocess output of model.
    """

    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")

    data = np.load(fname)

    return _dequantize(np.mean(data, axis=0, keepdims=True).astype(np.float32)[0], dtype=dtype)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process input audio for GenreNet model')
    parser.add_argument('--show', action='store_true', help='Show the preprocessed audio')
    parser.add_argument('--save-all', action='store_true', help='Path to save the preprocessed output')
    parser.add_argument('--mel', action='store_true', help='Convert input to mel spectrogram during preprocessing')
    args = parser.parse_args()

    def _display_input(data_chunks, **kwargs):

        mel = kwargs.get("mel", False)
        
        num_to_show = 8
        plt.figure(figsize=(12, 8))

        for i in range(num_to_show):

            plt.subplot(2, 4, i+1)

            try:
                if mel:
                    data_chunk = data_chunks[i].reshape(128, 128)
                else:
                    data_chunk = data_chunks[i]
                    data_chunk = melspectrogram(y=data_chunk, sr=22050, hop_length=512, n_fft=2048).T
                    data_chunk = data_chunk[:-1 * (data_chunk.shape[0] % 128)]
                chunk_dB = librosa.power_to_db(data_chunk.T, ref=np.max)
                plt.imshow(chunk_dB, aspect='auto', origin='lower')
                plt.title(f"Window {i}")
                plt.axis('off')
            except:
                continue

        plt.tight_layout()
        plt.show()

    def _display_output(output_inference):

        for i, label in enumerate(GENRES):
            print(f"  {label}:  {(output_inference[i] ):.4f}")

    SAMPLE_INPUT = "./sample_data/test.wav"
    SAMPLE_OUTPUT = "./sample_data/sample_output.npy"
    
    # Preprocess input to create bitstream
    if args.mel:
        x = preprocess(SAMPLE_INPUT, mel=True)
    else:
        x = preprocess(SAMPLE_INPUT)
    print(f"Preprocessed as input tensor with shape {x.shape} and type {x.dtype}.")

    # Show preprocessed data chunks from input audio
    if args.show:
        if args.mel:
            _display_input(x, mel=True)
        else:
            _display_input(x)

    # Postprocess output to get human-readable results
    y = postprocess(SAMPLE_OUTPUT)
    print(f"Postprocessed output with shape {y.shape} and type {y.dtype}.")

    if args.show:
        _display_output(y)

    # Save all intermediate files
    if args.save_all:
        np.save('sample_preprocessed.npy', x)
        np.save('sample_postprocessed.npy', y)
