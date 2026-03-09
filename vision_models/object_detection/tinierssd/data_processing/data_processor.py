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

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import data_processing.utils.object_detection_utils as obj_detect_utils

INPUT_TENSOR_SHAPE = (120, 160)

class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

def _pack_bitstream(input):

    merged_array = (
        (input[2].astype(np.uint32) & 0xFF) << 16 |   # B
        (input[1].astype(np.uint32) & 0xFF) << 8 |    # G
        (input[0].astype(np.uint32) & 0xFF)           # R
    )

    merged_array = merged_array[np.newaxis, ...]  # shape (1, 120, 160)
    flat = merged_array.flatten()

    return flat

def _unpack_bitstream(input_bitstream):

    b = (input_bitstream >> 16) & 0xFF
    g = (input_bitstream >> 8) & 0xFF
    r = input_bitstream & 0xFF

    rgb = np.stack([r, g, b], axis=1)
    rgb = rgb.astype(np.int8)
    img = rgb.astype(np.int16).reshape(INPUT_TENSOR_SHAPE[0], INPUT_TENSOR_SHAPE[1], 3) + 128

    return img
    
def preprocess(fname, pack=True, **kwargs):
    """
    Preprocess input of model.
    """
    
    img = Image.open(fname).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(INPUT_TENSOR_SHAPE),
        transforms.ToTensor(),
        normalize()
    ])

    img_tensor = transform(img)
    
    if pack:
        return _pack_bitstream(img_tensor.numpy())
    else:
        return img_tensor.numpy()

def postprocess(fname, **kwargs):
    """
    Postprocess output of model.
    """

    if not fname.endswith('.npy'):
        raise ValueError("Unsupported file format. Please provide a .npy file.")
    
    # Load and scale numpy file
    data = np.load(fname)

    if data.shape != (10836,) and data.shape != (1, 10836):
        raise ValueError(f"Unsupported file dimensions. Provided .npy file of shape (10836,) or (1, 10836) instead of {data.shape}.")
    
    # Split data into int8 values
    data_int8 = data.view(np.uint16) >> 3
    data_int8 = data_int8.astype(np.int8)
    
    # Individual shapes of all outputs
    locs_size = np.array([[16, INPUT_TENSOR_SHAPE[0]//8, INPUT_TENSOR_SHAPE[1]//8 ],
        [16, INPUT_TENSOR_SHAPE[0]//16, INPUT_TENSOR_SHAPE[1]//16 ],
        [16, INPUT_TENSOR_SHAPE[0]//32, INPUT_TENSOR_SHAPE[1]//32 ],
        [16, INPUT_TENSOR_SHAPE[0]//64, INPUT_TENSOR_SHAPE[1]//64 ] ])

    kpts_size = np.array([[32, INPUT_TENSOR_SHAPE[0]//8, INPUT_TENSOR_SHAPE[1]//8 ],
            [32, INPUT_TENSOR_SHAPE[0]//16, INPUT_TENSOR_SHAPE[1]//16 ],
            [32, INPUT_TENSOR_SHAPE[0]//32, INPUT_TENSOR_SHAPE[1]//32 ],
            [32, INPUT_TENSOR_SHAPE[0]//64, INPUT_TENSOR_SHAPE[1]//64 ] ])

    scores_size = np.array([[8, INPUT_TENSOR_SHAPE[0]//8, INPUT_TENSOR_SHAPE[1]//8 ],
            [8, INPUT_TENSOR_SHAPE[0]//16, INPUT_TENSOR_SHAPE[1]//16 ],
            [8, INPUT_TENSOR_SHAPE[0]//32, INPUT_TENSOR_SHAPE[1]//32 ],
            [8, INPUT_TENSOR_SHAPE[0]//64, INPUT_TENSOR_SHAPE[1]//64 ] ])

    all_sizes = [locs_size, kpts_size, scores_size]

    locs_total = np.prod(locs_size, axis=1).sum()
    kpts_total = np.prod(kpts_size, axis=1).sum()

    partitioned_data = [ data_int8[0: locs_total],
                         data_int8[locs_total: locs_total + kpts_total],
                         data_int8[locs_total + kpts_total: ]]
    
    batch_size = 1
    n_classes = 2
    shape = [4, 8, n_classes]

    reshaped_output = []
    for i in range(3):

        a = 0
        fire8 =  partitioned_data[i][a: a + np.prod(all_sizes[i][0])]
        fire8 = fire8.reshape(all_sizes[i][0])
        fire8 = torch.as_tensor(fire8/16, dtype=torch.float32).unsqueeze(0)
        fire8 = fire8.permute(0, 2, 3, 1).contiguous()
        fire8 = fire8.view(batch_size, -1, (shape[i]))

        a = a + np.prod(all_sizes[i][0])
        fire9 =  partitioned_data[i][a: a + np.prod(all_sizes[i][1])]
        fire9 = fire9.reshape(all_sizes[i][1])
        fire9 = torch.as_tensor(fire9/16, dtype=torch.float32).unsqueeze(0)
        fire9 = fire9.permute(0, 2, 3, 1).contiguous()
        fire9 = fire9.view(batch_size, -1, (shape[i]))

        a = a + np.prod(all_sizes[i][1])
        fire10 = partitioned_data[i][a: a + np.prod(all_sizes[i][2])]
        fire10 = fire10.reshape(all_sizes[i][2])
        fire10 = torch.as_tensor(fire10/16, dtype=torch.float32).unsqueeze(0)
        fire10 = fire10.permute(0, 2, 3, 1).contiguous()
        fire10 = fire10.view(batch_size, -1, (shape[i]))

        a = a + np.prod(all_sizes[i][2])
        conv12 = partitioned_data[i][a: a + np.prod(all_sizes[i][3])]
        conv12 = conv12.reshape(all_sizes[i][3])
        conv12 = torch.as_tensor(conv12/16, dtype=torch.float32).unsqueeze(0)
        conv12 = conv12.permute(0, 2, 3, 1).contiguous()
        conv12 = conv12.view(batch_size, -1, (shape[i]))

        reshaped_output.append([fire8, fire9, fire10, conv12])

    locs = torch.cat(reshaped_output[0], dim=1)
    kpts = torch.cat(reshaped_output[1], dim=1)
    locs = torch.cat([locs, kpts], dim=2)
    classes_scores = torch.cat(reshaped_output[2], dim=1)

    min_score = kwargs.get('min_score', 0.4)
    max_overlap = kwargs.get('max_overlap', 0.1)
    top_k = kwargs.get('top_k', 20)

    prior_boxes = obj_detect_utils.create_prior_boxes()
    all_images_boxes, all_images_kpts, all_images_labels, all_images_scores = \
        obj_detect_utils.detect_objects(locs, classes_scores, prior_boxes,
                            min_score=min_score,
                            max_overlap=max_overlap,
                            top_k=top_k,
                            return_kpts=True)
    
    return all_images_boxes, all_images_kpts, all_images_labels, all_images_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process image for TinierSSD model')
    parser.add_argument('--show', action='store_true', help='Show the preprocessed image and bounding boxes')
    parser.add_argument('--save-all', action='store_true', help='Path to save the preprocessed output')
    args = parser.parse_args()
    
    def _display_input(img_bitstream):

        img_to_plot = _unpack_bitstream(img_bitstream)
        plt.imshow(img_to_plot)
        plt.show()


    def _display_output(img_bitstream, boxes, kpts):

        fig, ax = plt.subplots(1)

        img_to_plot = _unpack_bitstream(img_bitstream)
        ax.imshow(img_to_plot)

        scale = np.array([INPUT_TENSOR_SHAPE[1], INPUT_TENSOR_SHAPE[0], INPUT_TENSOR_SHAPE[1], INPUT_TENSOR_SHAPE[0]])

        boxes_resized = [
            box.detach().cpu().numpy() * scale for box in boxes
        ]

        for b in range(len(boxes_resized[0])):
            bb = boxes_resized[0][b]
            rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=3,
                                    edgecolor='b', facecolor="none")
            ax.add_patch(rect)

        keypoints_scaled = []
        for kp in kpts[0]:
            kp = kp.detach().cpu().numpy()
            kp_scaled = np.array([
                bb[0] + kp[0] * (bb[2]-bb[0]), bb[1] + kp[1] * (bb[3]-bb[1]),
                bb[0] + kp[2] * (bb[2]-bb[0]), bb[1] + kp[3] * (bb[3]-bb[1]),
                bb[0] + kp[4] * (bb[2]-bb[0]), bb[1] + kp[5] * (bb[3]-bb[1]),
                bb[0] + kp[6] * (bb[2]-bb[0]), bb[1] + kp[7] * (bb[3]-bb[1])
            ])
            keypoints_scaled.append(kp_scaled)

        for kp in keypoints_scaled:
            x = kp[0::2]
            y = kp[1::2]
            ax.scatter(x, y, c='r', s=30)

        plt.show()

    SAMPLE_INPUT = "./sample_data/sample_input.png"
    SAMPLE_OUTPUT = "./sample_data/sample_output.npy"
    
    # Preprocess input to create bitstream
    x = preprocess(SAMPLE_INPUT)
    print(f"Preprocessed bitstream shape: {x.shape}. Stream length in bytes of four: {x.shape[0]}.")

    # Show preprocessed image from created bitstream
    if args.show:
        _display_input(img_bitstream=x)

    # Postprocess output to get bounding boxes
    boxes, kpts, _, _ = postprocess(SAMPLE_OUTPUT, min_score=0.4, max_overlap=0.1, top_k=20)
    print(f"Postprocessed output with {len(boxes)} object/s detected.")

    # Show detected objects
    if args.show:
        _display_output(img_bitstream=x, boxes=boxes, kpts=kpts)

    # Save preprocessed bitstream
    if args.save_all:
        np.save('sample_preprocessed.npy', x)
