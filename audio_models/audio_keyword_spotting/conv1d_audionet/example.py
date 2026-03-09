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
from pathlib import Path
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference
from data_processing.data_processor import preprocess

NUM_CATEGORIES   = 21
CATEGORY_LABELS = [
    'up', 'down', 'left', 'right', 'stop', 'go',
    'yes', 'no', 'on', 'off',
    'one', 'two', 'three', 'four', 'five',
    'six', 'seven', 'eight', 'nine', 'zero',
    '_unknown_'
]

def predict_keyword(features, speech_model):
    output = speech_model.generate_expected_output(features)
    logits = output[0].detach().numpy().astype(np.float32)
    logits = logits / 1000.0
    logits -= logits.max()
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs, output

def run_inference(wav_path, ai8x_model, output_folder):
    speech_model = AI8XInference(
        model_class="ai87kws20netv3",
        model_module="ai87net-kws20-v3",
        model_checkpoint_path=Path(ai8x_model),
        hardware="MAX78002"
    )

    features = preprocess(wav_path, pack=False)
    category_probabilities, raw_output = predict_keyword(features, speech_model)
    predicted_idx      = np.argmax(category_probabilities)
    predicted_category = CATEGORY_LABELS[predicted_idx]
    confidence         = category_probabilities[predicted_idx]

    print("Prediction Results:")
    for i, (label, prob) in enumerate(zip(CATEGORY_LABELS, category_probabilities)):
        marker = "***" if i == predicted_idx else "   "
        print(f"{marker} {label:10s}: {prob:.4f}")
    print(f"Detected keyword: '{predicted_category}' (confidence: {confidence:.4f})")

    # Save prediction summary as .txt
    output_file = os.path.join(output_folder, "prediction.txt")
    with open(output_file, 'w') as f:
        f.write(f"Predicted Category: {predicted_category}\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write(f"\nAll Probabilities:\n")
        for label, prob in zip(CATEGORY_LABELS, category_probabilities):
            f.write(f"{label}: {prob:.4f}\n")

    return predicted_category, confidence

def main(argv):
    root_path     = os.path.dirname(__file__)
    ai8x_path     = os.path.join(root_path, "data", "model", "ai87-kws20_v3-qat8-q.pth.tar")
    output_folder = os.path.join(root_path, "data", "output")
    audio_path    = argv[0] if argv else os.path.join(root_path, "data", "input", "yes.wav")
    os.makedirs(output_folder, exist_ok=True)

    run_inference(audio_path, ai8x_path, output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])
