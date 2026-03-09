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

from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

class pack_uint8_to_uint32:
    """
    Pack uint8 array into uint32 array for max78002 input.
    """
    def __init__(self):
        pass

    def __call__(self, input):
        input = input.long()
        
        merged_array = (
            ((input[2] & 0xFF) << 16) |
            ((input[1] & 0xFF) << 8) |
            (input[0] & 0xFF)
        )

        merged_array = merged_array.unsqueeze(0)  # shape (1, 32, 32)

        return merged_array.flatten()

def preprocess(fname, pack=True, **kwargs):
    INPUT_TENSOR_SHAPE = (32, 32)
    
    img = Image.open(fname).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(INPUT_TENSOR_SHAPE),
        transforms.ToTensor(),
        normalize(),
        pack_uint8_to_uint32() if pack else transforms.Lambda(lambda x: x)
    ])

    img_tensor = transform(img)

    return img_tensor.numpy()

def postprocess(fname, **kwargs):
    """
    Postprocess output of model.
    """
    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")

    # Load and scale numpy file
    return np.load(fname).astype(np.float32) / 1000
        

if __name__ == "__main__":
    # Setup argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Process images for visual wake word model')
    parser.add_argument('--show', action='store_true', help='Show the preprocessed image')
    parser.add_argument('--save-all', action='store_true', help='Set to true to save all processed data')
    args = parser.parse_args()

    # Sample data paths
    SAMPLE_INPUT = "./sample_data/bottle.png"
    SAMPLE_OUTPUT = "./sample_data/sample_output.npy"

    if args.show:
        import matplotlib.pyplot as plt
        img = Image.open(SAMPLE_INPUT).convert("RGB")
        img.show()
        img_tensor = preprocess(SAMPLE_INPUT, pack=False)
        img_scaled = img_tensor / 128 # Scale to int8
        img_np = np.transpose(img_scaled, (1, 2, 0))
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()

    # Preprocess input to create bitstream
    x = preprocess(SAMPLE_INPUT)
    print(f"Preprocessed bitstream shape: {x.shape}. Stream length in bytes of four: {x.shape[0]}.")

    # Postprocess output to get class confidence distribution
    y = postprocess(SAMPLE_OUTPUT)
    print(f"One-hot-encoded class confidence: \n {y}")

    # Save all test data
    if args.save_all:
        np.save('sample_preprocessed.npy', x)
        np.save('sample_postprocessed.npy', y)
