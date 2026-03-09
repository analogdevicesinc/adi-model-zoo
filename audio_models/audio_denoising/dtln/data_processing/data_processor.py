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
import numpy as np
import soundfile as sf

SAMPLING_RATE = 16000
BLOCK_LEN = 512
BLOCK_SHIFT = 128
    
def preprocess(fname, **kwargs):
    """
    Preprocess input of model.
    """

    if not fname.endswith('.mp3') and not fname.endswith('.wav'):
        raise ValueError("Unsupported file format. Please provide a .mp3 or .wav file.")
    
    fft = kwargs.get("fft", False)

    audio, _ = librosa.load(fname, sr=SAMPLING_RATE)

    # Create array for all windows: audio, magnitude, phase
    all_audio = []
    all_fft_mag = []
    all_fft_phase = []

    # Create buffer
    in_buffer = np.zeros((BLOCK_LEN)).astype('float32')

    # Calculate number of blocks
    num_blocks = (audio.shape[0] - (BLOCK_LEN-BLOCK_SHIFT)) // BLOCK_SHIFT

    # Iterate over the number of blocks  
    for idx in range(num_blocks):
            
        # Shift values and write to buffer
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = audio[idx*BLOCK_SHIFT:(idx*BLOCK_SHIFT)+BLOCK_SHIFT]

        if fft:
            # Calculate fft of input block
            in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)

            # Reshape magnitude to input dimensions
            in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')

            all_fft_mag.append(in_mag)
            all_fft_phase.append(in_phase)
        else:    
            all_audio.append(in_buffer.copy())

    if fft:
        return np.array(all_fft_mag), np.array(all_fft_phase)
    else:
        return np.array(all_audio)

def postprocess(fname, **kwargs):
    """
    Postprocess output of model.
    """

    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")

    data = np.load(fname)

    # Calculate length of audio
    audio_length = (data.shape[0]*BLOCK_SHIFT) + (BLOCK_LEN-BLOCK_SHIFT)

    # Create buffer
    out_buffer = np.zeros((BLOCK_LEN)).astype('float32')
    denoised_audio = np.zeros(audio_length).astype('float32')

    for idx, out_block in enumerate(data):

        # Shift values and write to buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros((BLOCK_SHIFT))
        out_buffer  += np.squeeze(out_block)

        # Write block to output file
        denoised_audio[idx*BLOCK_SHIFT:(idx*BLOCK_SHIFT)+BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]

    return denoised_audio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process input audio for DTLN model')
    parser.add_argument('--save-all', action='store_true', help='Path to save the preprocessed and postprocessed output')
    parser.add_argument('--fft', action='store_true', help='Preprocess input audio into FFT magnitude and phase')    
    args = parser.parse_args()

    def _save_audio(output_path, audio_data):

        sf.write(output_path, audio_data, SAMPLING_RATE)

    SAMPLE_INPUT = "./sample_data/reading.wav"
    SAMPLE_OUTPUT = "./sample_data/reading_output.npy"
    
    # Preprocess input to extract features
    if args.fft:
        magnitude, phase = preprocess(SAMPLE_INPUT, fft=True)
        print(f"Preprocessed features as input tensor with shape {magnitude.shape} and type {magnitude.dtype}.")
    else:
        x = preprocess(SAMPLE_INPUT)
        print(f"Preprocessed features as input tensor with shape {x.shape} and type {x.dtype}.")

    # Postprocess output to get human-readable results
    denoised_audio = postprocess(SAMPLE_OUTPUT)
    print(f"Postprocessed output with shape {denoised_audio.shape} and type {denoised_audio.dtype}.")

    # Save all intermediate files
    if args.save_all:
        if args.fft:
            np.save('sample_preprocessed.npy', magnitude)
        else:
            np.save('sample_preprocessed.npy', x)
        _save_audio("denoised_output.wav", denoised_audio)
