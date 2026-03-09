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

import sys
import os
from pathlib import Path
from data_processing import data_processor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference

def preprocess(image_path):

    return data_processor.preprocess(image_path, pack=False).astype(np.float32)

def postprocess(data, fold_ratio):
    """
    Unfold data to reduce the number of channels. An interlaced approach used in this folding
    as explained in [1].
    [1] https://arxiv.org/pdf/2203.16528.pdf
    """

    num_out_channels = data.shape[1] // (fold_ratio*fold_ratio)

    img_batch_uf = torch.zeros((data.shape[0], num_out_channels,
                                data.shape[2]*fold_ratio, data.shape[3]*fold_ratio),
                            dtype=data.dtype, device=data.device, requires_grad=False)

    for i in range(fold_ratio):
        for j in range(fold_ratio):
            ch_index_start = num_out_channels*(i*fold_ratio + j)
            ch_index_end = num_out_channels * (i*fold_ratio + j + 1)
            img_batch_uf[:, :, i::fold_ratio, j::fold_ratio] = \
                data[:, ch_index_start:ch_index_end, :, :]

    return img_batch_uf.detach().numpy()

def save_output(mask, image, output_path):

    mask = np.expand_dims(mask[0][1], axis=0) + 128
    
    plt.figure(figsize=(6,6))
    plt.title("UNet Mask Inference")
    plt.imshow(image[0].transpose(1, 2, 0).astype(np.uint8) + 128)
    plt.imshow(mask.transpose(1, 2, 0), alpha=0.4, cmap="bwr")

    blue = patches.Patch(color='blue', label='subject')
    red = patches.Patch(color='red', label='background')

    plt.legend(handles=[blue, red])
    plt.savefig(output_path)

def mask_image(ai8x_model, image_path, output_folder):

    input_data = preprocess(image_path)

    model = AI8XInference(model_class="ai85unetlarge",
                        model_module="ai85net-unet",
                        model_checkpoint_path=Path(ai8x_model))
    
    # Generate inference
    mask = model.generate_expected_output(input_data)

    # Postprocess for display
    mask = postprocess(mask, 4)
    image = postprocess(torch.from_numpy(np.expand_dims(input_data, axis=0)), 4)

    save_output(mask, image, output_folder)
    print(f"Masking complete. Output saved to {output_folder}")

def main(argv):
    """
    Basic UNet ai8x Example
    Usage: python example.py <image file> [<image file> ...]
    """

    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    ai8x_path = os.path.join(root_path, "data", "model", "ai85-aisegment-unet-large-fakept-q.pth.tar")
    imgs = argv or [os.path.join(root_path, "data", "input", "sample_input.png")]

    for img_file in imgs:

        base_name = Path(img_file).stem + ".jpg"
        output_name = os.path.join(output_folder, "masked_" + base_name)
        
        mask_image(ai8x_path, img_file, output_name)

if __name__ == "__main__":
    main(sys.argv[1:])
