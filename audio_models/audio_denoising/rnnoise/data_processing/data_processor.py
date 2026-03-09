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

from utils.rnnoise_utils import RNNoisePreProcess

INPUT_QUANTIZATION_PARAMETERS = {
    'float32': {'scale': 1.0, 'zero_point': 0},
    'int8': {'scale': 0.22150060534477234, 'zero_point': 14},
}

OUTPUT_QUANTIZATION_PARAMETERS = {
    'float32': {'scale': 1.0, 'zero_point': 0},
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
        quantized = np.clip(quantized, -32768, 32767).astype(np.int16)
    elif dtype == 'int8':
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
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

SAMPLING_RATE = 48000
    
def preprocess(fname, **kwargs):
    """
    Preprocess input of model.
    """

    if not fname.endswith('.mp3') and not fname.endswith('.wav'):
        raise ValueError("Unsupported file format. Please provide a .mp3 or .wav file.")

    per_chunk_processing = RNNoisePreProcess(training=False)
    audio, _ = librosa.load(fname, sr=SAMPLING_RATE)
    audio = audio * (2 ** 15)

    num_samples = len(audio) // per_chunk_processing.FRAME_SIZE

    all_silence = []
    all_features = []
    all_X = []
    all_P = []
    all_Ex = []
    all_Ep = []
    all_Exp = []

    for i in range(num_samples):
        audio_window = audio[i * per_chunk_processing.FRAME_SIZE : (i + 1) * per_chunk_processing.FRAME_SIZE]
        silence, features, X, P, Ex, Ep, Exp = per_chunk_processing.process_frame(audio_window)
        features = np.expand_dims(features, (0, 1)).astype(np.float32)

        all_silence.append(silence)
        all_features.append(features)
        all_X.append(X)
        all_P.append(P)
        all_Ex.append(Ex)
        all_Ep.append(Ep)
        all_Exp.append(Exp)

    return _quantize(np.array(all_features), dtype=kwargs.get('dtype', 'int8')), np.array(all_silence), np.array(all_X), np.array(all_P), np.array(all_Ex), np.array(all_Ep), np.array(all_Exp)

def postprocess(fname, all_silence, all_X, all_P, all_Ex, all_Ep, all_Exp, **kwargs):
    """
    Postprocess output of model.
    """

    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")

    data = _dequantize(np.load(fname), dtype=kwargs.get('dtype', 'int8'))
    final_denoised_audio = []
    per_chunk_processing = RNNoisePreProcess(training=False)

    for i in range(data.shape[0]):

        denoise_output = data[i]
        denoise_output = np.squeeze(np.array(denoise_output))

        silence = all_silence[i]
        X = all_X[i]
        P = all_P[i]
        Ex = all_Ex[i]
        Ep = all_Ep[i]
        Exp = all_Exp[i]

        if not silence:
            denoised_audio_tmp = per_chunk_processing.post_process(silence, denoise_output, X, P, Ex, Ep, Exp)
            denoised_audio_tmp = np.rint(denoised_audio_tmp).astype(np.int16)
            final_denoised_audio.append(denoised_audio_tmp)
        else:
            final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))

    denoised_audio = np.concatenate(final_denoised_audio, axis=0)

    return denoised_audio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process input audio for RNNoise model')
    parser.add_argument('--save-all', action='store_true', help='Path to save the preprocessed and postprocessed output')
    args = parser.parse_args()

    def _save_audio(output_path, audio_data):

        sf.write(output_path, audio_data, SAMPLING_RATE, 'PCM_16')

    SAMPLE_INPUT = "./sample_data/reading.wav"
    SAMPLE_OUTPUT = "./sample_data/output.npy"
    
    # Preprocess input to extract features
    features, silence, X, P, Ex, Ep, Exp = preprocess(SAMPLE_INPUT)
    print(f"Preprocessed features as input tensor with shape {features.shape} and type {features.dtype}.")

    # Postprocess output to get human-readable results
    denoised_audio = postprocess(SAMPLE_OUTPUT, all_silence=silence, all_X=X, all_P=P, all_Ex=Ex, all_Ep=Ep, all_Exp=Exp)
    print(f"Postprocessed output with shape {denoised_audio.shape} and type {denoised_audio.dtype}.")

    # Save all intermediate files
    if args.save_all:
        np.save('sample_preprocessed.npy', features)
        _save_audio("denoised_output.wav", denoised_audio)
