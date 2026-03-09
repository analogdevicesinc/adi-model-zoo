# Copyright © 2026 Analog Devices, Inc. All Rights Reserved. 
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

import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from data_processing import data_processor

BLOCK_LEN = 512
BLOCK_SHIFT = 128
SAMPLING_RATE = 16000

def tflite_forward(interpreter_1, interpreter_2, input_data):
    """
    TFLite interpreter forward inference and return all outputs.
    """
    # Get the magnitude and phase inputs
    magnitude, phase = input_data

    # Get input and output tensors
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    # Create states for the LSTMs
    states_1 = np.zeros(input_details_1[0]['shape']).astype('float32')
    states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')

    # Calculate length of audio
    audio_length = (len(magnitude)*BLOCK_SHIFT) + (BLOCK_LEN-BLOCK_SHIFT)

    # Create output buffers
    out_buffer = np.zeros((BLOCK_LEN)).astype('float32')
    denoised_audio = np.zeros(audio_length).astype('float32')

    # Pass through input data
    for idx in range(len(magnitude)):

        in_mag = magnitude[idx]
        in_phase = phase[idx]

        # Set tensors to the first model
        interpreter_1.set_tensor(input_details_1[0]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[1]['index'], in_mag)

        # Run calculation
        interpreter_1.invoke()

        # Get the output of the first block
        out_mask = interpreter_1.get_tensor(output_details_1[0]['index']) 
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])  

        # Calculate the IFFT
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)

        # Reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')

        # Set tensors to the second block
        interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
        interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)

        # Run calculation
        interpreter_2.invoke()

        # Get output tensors
        out_block = interpreter_2.get_tensor(output_details_2[1]['index']) 
        states_2 = interpreter_2.get_tensor(output_details_2[0]['index']) 

        # Shift values and write to buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros((BLOCK_SHIFT))
        out_buffer  += np.squeeze(out_block)

        # Write block to output file
        denoised_audio[idx*BLOCK_SHIFT:(idx*BLOCK_SHIFT)+BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]

    return denoised_audio

def save_audio(audio, output_path):

    # Write to .wav file 
    sf.write(output_path, audio, SAMPLING_RATE) 

def denoise_audio_file(tflite_model_1, tflite_model_2, noisy_wav_path, output_folder):
    """
    Run a trained DTLN model over an audio file and save the output.
    """
    interpreter_1 = Interpreter(model_path=tflite_model_1)
    interpreter_2 = Interpreter(model_path=tflite_model_2)

    interpreter_1.allocate_tensors()
    interpreter_2.allocate_tensors()

    input_data = data_processor.preprocess(noisy_wav_path, fft=True)
    denoised_audio = tflite_forward(interpreter_1, interpreter_2, input_data)

    save_audio(denoised_audio, output_folder)
    print(f"Denoising complete. Output saved to {output_folder}")

def main(argv):
    """
    Basic DTLN TFLite Example
    Usage: python example.py <audio file> [<audio file> ...]
    """

    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    tflite_path_1 = os.path.join(root_path, "data", "model", "model_float32", "model_float_1.tflite")
    tflite_path_2 = os.path.join(root_path, "data", "model", "model_float32", "model_float_2.tflite")
    wavs = argv or [os.path.join(root_path, "data", "input", "reading.wav")]

    for wav_file in wavs:

        base_name = Path(wav_file).stem + ".wav"
        output_name = os.path.join(output_folder, "denoised_" + base_name)
        denoise_audio_file( tflite_model_1=tflite_path_1,
                            tflite_model_2=tflite_path_2,
                            noisy_wav_path=wav_file,
                            output_folder=output_name)

if __name__ == "__main__":
    main(sys.argv[1:])
