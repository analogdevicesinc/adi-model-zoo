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

import numpy as np
import torch
import librosa

SAMPLE_RATE = 16000

def _quantize_audio(data, num_bits=8, compand=False, mu=10):
    """
    Quantize audio
    """
    step_size = 2.0 / 2**num_bits
    max_val = 2**num_bits - 1
    q_data = np.round((data - (-1.0)) / step_size)
    q_data = np.clip(q_data, 0, max_val)
    return np.uint8(q_data)

def preprocess(fname, pack=True, **kwargs):
    """
    Preprocess audio file for max78000 into bitstream.
    """
    # Signal extraction parameters
    index = kwargs.get('index', 0)
    scale = kwargs.get('scale', 1.0)

    # Read audio file
    if not fname.endswith('.mp3') and not fname.endswith('.wav'):
        raise ValueError("Unsupported file format. Please provide a .mp3 or .wav file.")
    
    sig, _ = librosa.load(fname, sr=SAMPLE_RATE)
    sig = np.clip(sig, -1.0, 1.0)
    sig = (sig * 32768)
    sig = np.round(sig).astype(np.int16)

    # Trim or pad signal to 16384 samples
    sig = scale*sig[index:index+16384]
    sig = np.pad(sig, [0, 16384-sig.size])
    
    sig_q = _quantize_audio(sig/(2*max(sig)))  # double the magnitude

    samples = np.zeros((128, 128)).astype(np.float32)
    for k in range(128):
        for j in range(128):
            samples[j, k] = sig_q[k*128 + j]

    samples = torch.from_numpy(samples) - 128.0
    if not pack:
        return samples.detach().numpy()

    samples = torch.unsqueeze(samples, 0)
    sample_in = samples.detach().numpy().astype(np.uint8)

    # Flatten into bitstream
    x = sample_in.reshape(2, 64, 128)
    x = x.reshape(2, 16, 4, 128)
    x = np.transpose(x, (1, 3, 0, 2))
    x = np.flip(x, 3)
    x = x.reshape(16, 256, 4)
    x = x.reshape(4096, 4)

    # Merge uint8 into uint32
    merged = np.zeros(4096).astype(np.uint32)

    for i in range(4096):
        merged[i] = (((x[i][0]) & 0xFF) << 24) | \
                    (((x[i][1]) & 0xFF) << 16 ) | \
                    (((x[i][2]) & 0xFF) << 8 ) | \
                    (((x[i][3]) & 0xFF))

    return merged

def postprocess(fname, **kwargs):
    """
    Postprocess output of model.
    """
    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")

    # Load and scale numpy file
    return np.load(fname).astype(np.float32) / 1000


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process images for visual wake word model')
    parser.add_argument('--save-all', action='store_true', help='Set to true to save all processed data')
    args = parser.parse_args()

    # Sample data paths
    SAMPLE_INPUT = "./sample_data/stop.wav"
    SAMPLE_OUTPUT = "./sample_data/sample_output.npy"

    # Classes
    LABELS = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
              'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', '_unk_'] # Last idx is _unknown_

    # Preprocess input to create bitstream
    x = preprocess(SAMPLE_INPUT, pack=False)
    print(f"Preprocessed bitstream shape: {x.shape}. Stream length in bytes of four: {x.shape[0]}.")

    # Postprocess output to create class confidence distribution
    y = postprocess(SAMPLE_OUTPUT)
    print(f"Confidence distribution:")
    for label, confidence in zip(LABELS,y):
        print(f"\t{label}: \t{confidence * 100:.02f}%")

    # Save all test data
    if args.save_all:
        np.save('sample_preprocessed.npy', x)
        np.save('sample_postprocessed.npy', y)
