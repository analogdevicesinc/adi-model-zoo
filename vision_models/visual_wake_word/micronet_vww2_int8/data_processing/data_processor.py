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
#/* Portions Copyright (c) 2026 Analog Devices, Inc. */

from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F

class Quantize:
    """
    Quantize output using model scale factor and zero point.
    """
    SCALE = 0.007549178320914507
    ZERO_POINT = -72

    def __init__(self):
        pass

    def __call__(self, tensor):
        return (tensor.div(self.SCALE).round().add(self.ZERO_POINT).clamp(min=-128, max=127)).to(torch.int8)

class Dequantize:
    """
    Dequantize output using model scale factor and zero point.
    """
    SCALE = 0.05457817763090134
    ZERO_POINT = 34

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.float32).sub(self.ZERO_POINT).mul(self.SCALE)
    
class Softmax:
    def __init__(self, dim=-1):
        self.dim = dim  # dimension to apply softmax

    def __call__(self, tensor):
        # Apply softmax along the specified dimension
        return F.softmax(tensor, dim=self.dim)


def preprocess(fname, **kwargs):
    INPUT_TENSOR_SHAPE = (50, 50)
    
    img = Image.open(fname).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(INPUT_TENSOR_SHAPE),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda x: x.unsqueeze(-1)),   # adds the final dimension
        Quantize() if kwargs.get('quantize', True) else transforms.Lambda(lambda x: x)
    ])

    img_tensor = transform(img)

    return img_tensor.numpy()

def postprocess(fname, **kwargs):
    """
    Postprocess output of model.
    """
    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")

    transform = transforms.Compose([
        Dequantize() if kwargs.get('dequantize', True) else transforms.Lambda(lambda x: x),
        Softmax(),
    ])

    data = np.load(fname)     # Load numpy files
    data = torch.from_numpy(data)     # Convert to torch tensor
    conf = transform(data)     #  Apply dequantization and softmax

    return conf.numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process images for visual wake word model')
    parser.add_argument('--show', action='store_true', help='Show the preprocessed image')
    parser.add_argument('--save-all', action='store_true', help='Set to true to save all processed data')
    args = parser.parse_args()

    def _show(img_path, tensor):
        import matplotlib.pyplot as plt
        # Show image vs preprocessed tensor
        img = Image.open(img_path).convert("RGB")
        img.show()
        img_gray = x.squeeze() # remove last dimension
        plt.imshow(img_gray, cmap="gray")
        plt.axis("off")
        plt.show()
        if args.save_all:
            plt.savefig('preprocessed_image.png')

    SAMPLE_INPUT = "./sample_data/person.jpg"
    SAMPLE_OUTPUT = "./sample_data/sample_output.npy"
    
    # Preprocess input to create bitstream
    x = preprocess(SAMPLE_INPUT, show=True)
    # Show actual image vs preprocessed
    if args.show:
        _show(SAMPLE_INPUT, x)
    print(f"Preprocessed as input tensor with shape {x.shape} and type {x.dtype}.")

    # Postprocess output to get human-readable results
    y = postprocess(SAMPLE_OUTPUT)
    labels = ['no person', 'person']
    print(f"Postprocessed output with shape {y.shape} and type {y.dtype}.")
    for i, label in enumerate(labels):
        print(f"  {label}: {y[i]:.6f}")

    # Save all intermediate files
    if args.save_all:
        np.save('sample_preprocessed.npy', x)
        np.save('sample_postprocessed.npy', y)
