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
import librosa
import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter
from data_processing.utils.rnnoise_utils import RNNoisePreProcess

SAMPLING_RATE = 48000

def tflite_forward(interpreter, input_data):
    """
    TFLite interpreter forward inference and return all outputs.
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

def save_audio(audio, output_path):

    # Write to .wav file
    sf.write(output_path, audio, SAMPLING_RATE, 'PCM_16')

def denoise_audio_file(tflite_model, noisy_audio_path, output_folder):
    """
    Run a trained RNNoise model over an audio file and save the output.
    """
    interpreter = Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    vad_gru_state = np.zeros((1, 24), dtype=np.float32)
    noise_gru_state = np.zeros((1, 48), dtype=np.float32)
    denoise_gru_state = np.zeros((1, 96), dtype=np.float32)

    preprocess = RNNoisePreProcess(training=False)
    audio, _ = librosa.load(noisy_audio_path, sr=SAMPLING_RATE)
    audio = audio * (2 ** 15)

    final_denoised_audio = []
    num_samples = len(audio) // preprocess.FRAME_SIZE

    for i in range(num_samples):
        audio_window = audio[i * preprocess.FRAME_SIZE : (i + 1) * preprocess.FRAME_SIZE]
        silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
        features = np.expand_dims(features, (0, 1)).astype(np.float32)

        if not silence:
            input_data = [features, vad_gru_state, noise_gru_state, denoise_gru_state]
            model_output = tflite_forward(interpreter, input_data)
            denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model_output
            vad_gru_state = np.squeeze(vad_gru_state, axis=1)
            noise_gru_state = np.squeeze(noise_gru_state, axis=1)
            denoise_gru_state = np.squeeze(denoise_gru_state, axis=1)
            denoise_output = np.squeeze(np.array(denoise_output))
            denoised_audio_tmp = preprocess.post_process(silence, denoise_output, X, P, Ex, Ep, Exp)
            denoised_audio_tmp = np.rint(denoised_audio_tmp).astype(np.int16)
            final_denoised_audio.append(denoised_audio_tmp)
        else:
            final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))

    denoised_audio = np.concatenate(final_denoised_audio, axis=0)

    save_audio(denoised_audio, output_folder)
    print(f"Denoising complete. Output saved to {output_folder}")

def main(argv):
    """
    Basic RNNoise TFLite Example
    Usage: python example.py <audio file> [<audio file> ...]
    """
    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    tflite_path = os.path.join(root_path, "data", "model", "rnnoise_INT8.tflite")
    wavs = argv or [os.path.join(root_path, "data", "input", "reading.wav")]

    for wav_file in wavs:

        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        output_path = os.path.join(output_folder, f"{base_name}_denoised.wav")
        denoise_audio_file(tflite_path, wav_file, output_path)

if __name__ == "__main__":
    main(sys.argv[1:])
