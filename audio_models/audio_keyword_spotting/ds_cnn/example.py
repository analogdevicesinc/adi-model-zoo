#!/usr/bin/env python3
# Copyright © 2025-2026 AI Edge Model Catalog. All rights reserved.
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
import struct
import wave
from data_processing.data_processor import preprocess
import numpy as np
import ai_edge_litert.interpreter as litert
import librosa

SAMPLE_RATE      = 16000
WINDOW_SIZE_MS   = 30
WINDOW_STRIDE_MS = 20
FEATURE_SHAPE    = (49, 10)
NUM_CATEGORIES   = 4
CATEGORY_LABELS  = ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
NUM_CHANNELS    = 10
LOWER_EDGE_HZ   = 125.0
UPPER_EDGE_HZ   = 7500.0
INPUT_SCALE      = 0.10171568393707275
INPUT_ZERO_POINT = -128
LEGACY_OUTPUT_SCALING = 25.6

def load_wav_file(wav_path):
    with wave.open(wav_path, 'rb') as wav_file:
        sample_rate  = wav_file.getframerate()
        num_frames   = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        audio_data = wav_file.readframes(num_frames)

        if sample_width == 2:
            audio_samples = np.array(struct.unpack(f'{num_frames * num_channels}h', audio_data), dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if num_channels == 2:
            audio_samples = audio_samples[::2]

        return audio_samples, sample_rate

def preprocess_audio_window_librosa(audio_window, use_int8_output=True):
    audio_float = audio_window.astype(np.float32) / 32768.0

    mel_spec = librosa.feature.melspectrogram(
        y=audio_float,
        sr=SAMPLE_RATE,
        n_fft=480,
        hop_length=len(audio_float),
        n_mels=NUM_CHANNELS,
        fmin=LOWER_EDGE_HZ,
        fmax=UPPER_EDGE_HZ,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel_spec, ref=1.0, top_db=None)
    feature = log_mel[:, 0]
    feature = feature + 80.0
    feature = feature * (LEGACY_OUTPUT_SCALING / 80.0)
    feature = np.clip(feature, 0.0, LEGACY_OUTPUT_SCALING)

    if use_int8_output:
        feature_quantized = feature / INPUT_SCALE + INPUT_ZERO_POINT
        feature = np.clip(feature_quantized, -128, 127).astype(np.int8)
    else:
        feature = feature.astype(np.float32)

    return feature

def generate_features(audio_samples, use_int8_output=True):
    window_size   = int(WINDOW_SIZE_MS * SAMPLE_RATE / 1000)
    window_stride = int(WINDOW_STRIDE_MS * SAMPLE_RATE / 1000)

    features = np.zeros(FEATURE_SHAPE, dtype=np.int8 if use_int8_output else np.float32)

    if len(audio_samples) < SAMPLE_RATE:
        audio_samples = np.pad(audio_samples, (0, SAMPLE_RATE - len(audio_samples)), mode='constant')

    start_index = 0
    for frame_number in range(FEATURE_SHAPE[0]):
        end_index = start_index + window_size
        if end_index > len(audio_samples):
            window = np.pad(audio_samples[start_index:], (0, end_index - len(audio_samples)), mode='constant')
        else:
            window = audio_samples[start_index:end_index]

        features[frame_number] = preprocess_audio_window_librosa(window, use_int8_output)
        start_index += window_stride

    return features

def predict_keyword(features, speech_interpreter):
    input_details  = speech_interpreter.get_input_details()[0]
    output_details = speech_interpreter.get_output_details()[0]

    flattened_features = features.flatten().reshape([1, -1])

    if input_details['dtype'] == np.int8 and features.dtype != np.int8:
        flattened_features = flattened_features / INPUT_SCALE + INPUT_ZERO_POINT
        flattened_features = np.clip(flattened_features, -128, 127).astype(np.int8)
    elif input_details['dtype'] != np.int8 and features.dtype == np.int8:
        flattened_features = (flattened_features.astype(np.float32) - INPUT_ZERO_POINT) * INPUT_SCALE
    else:
        flattened_features = flattened_features.astype(input_details['dtype'])

    speech_interpreter.set_tensor(input_details['index'], flattened_features)
    speech_interpreter.invoke()

    output = speech_interpreter.get_tensor(output_details['index'])

    if output_details['dtype'] != np.float32:
        output_scale, output_zero_point = output_details['quantization']
        output = output_scale * (output.astype(np.float32) - output_zero_point)

    return output[0]

def run_inference(wav_path, speech_model_path, output_folder):
    audio_samples, sample_rate = load_wav_file(wav_path)
    print(f"Audio loaded: {len(audio_samples)} samples at {sample_rate} Hz")

    speech_interpreter = litert.Interpreter(model_path=speech_model_path)
    speech_interpreter.allocate_tensors()

    features = preprocess(wav_path)

    category_probabilities = predict_keyword(features, speech_interpreter)

    predicted_idx      = np.argmax(category_probabilities)
    predicted_category = CATEGORY_LABELS[predicted_idx]
    confidence         = category_probabilities[predicted_idx]

    print("Prediction Results:")
    for i, (label, prob) in enumerate(zip(CATEGORY_LABELS, category_probabilities)):
        marker = "***" if i == predicted_idx else "   "
        print(f"{marker} {label:10s}: {prob:.4f}")
    print(f"Detected keyword: '{predicted_category}' (confidence: {confidence:.4f})")

    output_file = os.path.join(output_folder, "prediction.txt")
    with open(output_file, 'w') as f:
        f.write(f"Predicted Category: {predicted_category}\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write(f"\nAll Probabilities:\n")
        for label, prob in zip(CATEGORY_LABELS, category_probabilities):
            f.write(f"{label}: {prob:.4f}\n")

    return predicted_category, confidence

def main(argv):
    root_path    = os.path.dirname(__file__)
    model_path   = os.path.join(root_path, "data", "model", "ds_cnn_s_float32.tflite")
    output_folder = os.path.join(root_path, "data", "output")
    audio_path   = argv[0] if argv else os.path.join(root_path, "data", "input", "yes.wav")
    os.makedirs(output_folder, exist_ok=True)
    run_inference(audio_path, model_path, output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])
