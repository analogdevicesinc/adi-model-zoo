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
import matplotlib.pyplot as plt

INPUT_TENSOR_SHAPE = (192, 192)
FOLD_RATIO = 4
NUM_CLASSES = 2

class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

class fold:
    """
    Fold data to increase the number of channels. An interlaced approach used in this folding
    as explained in [1].
    [1] https://arxiv.org/pdf/2203.16528.pdf
    """
    def __init__(self, fold_ratio):
        self.fold_ratio = fold_ratio

    def __call__(self, img):
        if self.fold_ratio == 1:
            return img

        img_folded = None
        for i in range(self.fold_ratio):
            for j in range(self.fold_ratio):
                img_subsample = img[:, i::self.fold_ratio, j::self.fold_ratio]
                if img_folded is not None:
                    img_folded = torch.cat((img_folded, img_subsample), dim=0)
                else:
                    img_folded = img_subsample

        return img_folded

class unfold_batch:
    """
    Unfold data to reduce the number of channels. An interlaced approach used in this folding
    as explained in [1]. This operation is the reverse of the transformation implemented
    at ai8x.fold class.
    [1] https://arxiv.org/pdf/2203.16528.pdf
    """
    def __init__(self, fold_ratio):
        self.fold_ratio = fold_ratio

    def __call__(self, img_batch):
        if self.fold_ratio == 1:
            return img_batch

        num_out_channels = img_batch.shape[1] // (self.fold_ratio*self.fold_ratio)

        img_batch_uf = torch.zeros((img_batch.shape[0], num_out_channels,
                                    img_batch.shape[2]*self.fold_ratio, img_batch.shape[3]*self.fold_ratio),
                                dtype=img_batch.dtype, device=img_batch.device, requires_grad=False)

        for i in range(self.fold_ratio):
            for j in range(self.fold_ratio):
                ch_index_start = num_out_channels*(i*self.fold_ratio + j)
                ch_index_end = num_out_channels * (i*self.fold_ratio + j + 1)
                img_batch_uf[:, :, i::self.fold_ratio, j::self.fold_ratio] = \
                    img_batch[:, ch_index_start:ch_index_end, :, :]

        return img_batch_uf

def _pack_bitstream(input_bitstream):

    input_bitstream = input_bitstream.reshape(input_bitstream.shape[0]//FOLD_RATIO,
                          FOLD_RATIO,
                          INPUT_TENSOR_SHAPE[0]//FOLD_RATIO,
                          INPUT_TENSOR_SHAPE[1]//FOLD_RATIO)
    arr_u32 = input_bitstream.astype(np.uint32) & 0xFF

    merged_array = (
        (arr_u32[:, 3] << 24) |
        (arr_u32[:, 2] << 16) |
        (arr_u32[:, 1] << 8)  |
        (arr_u32[:, 0])
    )

    merged_array = merged_array[np.newaxis, ...]  # shape (1, 12, 48, 48)
    flat = merged_array.flatten()

    return flat

def _unpack_output(bitstream):

    data_int16 = bitstream.view(np.int16)
    data_int16 = data_int16.astype(np.int8)
    data_int16 = data_int16.reshape(NUM_CLASSES*FOLD_RATIO*FOLD_RATIO,
                                    INPUT_TENSOR_SHAPE[0]//FOLD_RATIO,
                                    INPUT_TENSOR_SHAPE[1]//FOLD_RATIO)
    
    return data_int16

def _unpack_input(bitstream):
    
    data_int8 = bitstream.view(np.int8)
    data_int8 = data_int8.reshape(12, 9216)
    data_int8 = data_int8.reshape(12, 2304, 4)
    data_int8 = data_int8.transpose(0, 2, 1)
    data_int8 = data_int8.reshape(48, 48, 48)

    return data_int8

def preprocess(fname, pack=True):

    img = Image.open(fname).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(INPUT_TENSOR_SHAPE),
        transforms.ToTensor(),
        normalize(),
        fold(FOLD_RATIO)
    ])

    img_tensor = transform(img)

    if pack:
        return _pack_bitstream(img_tensor.numpy())
    else:
        return img_tensor.numpy()

def postprocess(fname):
    """
    Postprocess output of model.
    """
    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")
    
    # Load and scale numpy file
    data = np.load(fname)

    if data.shape != (36864,) and data.shape != (1, 36864):
        raise ValueError(f"Unsupported file dimensions. Provided .npy file of shape (36864,) or (1, 36864) instead of {data.shape}.")

    data_int16 = _unpack_output(data)
    data_int16 = torch.from_numpy(data_int16).unsqueeze(0)

    transform = transforms.Compose([
        unfold_batch(FOLD_RATIO)
    ])

    img_mask = transform(data_int16)
    img_mask = img_mask.squeeze(0).cpu().numpy()

    return img_mask

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Process images for UNet model')
    parser.add_argument('--show', action='store_true', help='Show the input image and output mask')
    parser.add_argument('--save-all', action='store_true', help='Path to save the processed input and output')
    args = parser.parse_args()

    def _display_input(img):

        img_to_plot = _unpack_input(img)
        img_to_plot = torch.from_numpy(img_to_plot).unsqueeze(0)
        
        transform = transforms.Compose([
            unfold_batch(FOLD_RATIO)
        ])

        img_to_plot = transform(img_to_plot)
        img_to_plot = img_to_plot.squeeze(0).cpu().numpy()
        img_to_plot = img_to_plot.transpose(1, 2, 0)
        img_to_plot = img_to_plot.astype(np.uint16) + 128
        plt.imshow(img_to_plot)
        plt.show()

    def _display_output(img, img_mask):

        img_to_plot = _unpack_input(img)
        img_to_plot = torch.from_numpy(img_to_plot).unsqueeze(0)

        transform = transforms.Compose([
            unfold_batch(FOLD_RATIO)
        ])

        img_to_plot = transform(img_to_plot)
        img_to_plot = img_to_plot.squeeze(0).cpu().numpy()
        img_to_plot = img_to_plot.transpose(1, 2, 0)
        img_to_plot = img_to_plot.astype(np.uint16) + 128

        img_mask = img_mask.transpose(1, 2, 0).astype(np.uint16) + 128

        subject = np.clip(img_mask[:, :, 1:] + img_to_plot, 0, 255).astype(np.int16)
        background = np.clip(img_mask[:, :, 0:1] + img_to_plot, 0, 255).astype(np.int16)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(subject); ax[0].set_title("Image w/ Background Mask")
        ax[1].imshow(background); ax[1].set_title("Image w/ Subject Mask")
        plt.show()

    SAMPLE_INPUT = "./sample_data/sample_input.png"
    SAMPLE_OUTPUT = "./sample_data/sample_output.npy"
    
    # Preprocess input to create bitstream
    x = preprocess(SAMPLE_INPUT)
    print(f"Preprocessed bitstream shape: {x.shape}. Stream length in bytes of four: {x.shape[0]}.")

    if args.show:
        _display_input(x)

    # Postprocess output to get human-readable results (image masks)
    y = postprocess(SAMPLE_OUTPUT)
    print(f"Postprocessed output with {y.shape[0]} masks with dimensions {y[0].shape}.")

    if args.show:
        _display_output(img=x, img_mask=y)

    # Save all intermediate files
    if args.save_all:
        np.save('sample_preprocessed.npy', x)
        np.save('sample_postprocessed.npy', y)
