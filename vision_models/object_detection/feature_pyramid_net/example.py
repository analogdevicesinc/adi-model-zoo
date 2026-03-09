# Copyright (c) 2026 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from pathlib import Path
from data_processing import data_processor
from data_processing.utils import object_detection_utils as obj_detect_utils
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference

def preprocess(image_path):

    return data_processor.preprocess(image_path, pack=False).astype(np.float32)

def non_maximum_suppression(locs, scores, **kwargs):

    num_classes = 21
    min_score = kwargs.get("min_score", 0.3)
    max_overlap = kwargs.get("max_overlap", 0.3)
    top_k = kwargs.get("top_k", 50)

    # Decipher the locations and class scores to detect objects
    prior_boxes = obj_detect_utils.create_prior_boxes()
    all_images_boxes, all_images_labels, all_images_scores = obj_detect_utils.detect_objects(
                                                                locs, scores, num_classes, prior_boxes,
                                                                min_score, max_overlap, top_k)
    
    return all_images_boxes, all_images_labels, all_images_scores

def display_detection(image, all_images_boxes, all_images_labels, all_images_scores):

    img_to_plot = image
    img_to_plot = (img_to_plot + 128).astype(np.uint8)
    img_to_plot = img_to_plot.transpose([1, 2, 0])

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes.imshow(img_to_plot)

    subplot_title=("FeaturePyramidNet Output Inference")
    axes.set_title(subplot_title, fontsize = 14)

    # Predicted boxes & labels:
    boxes_resized = [[box[0] * img_to_plot.shape[1], \
                    box[1] * img_to_plot.shape[0], \
                    box[2] * img_to_plot.shape[1], \
                    box[3] * img_to_plot.shape[0]] for box in all_images_boxes[0].detach().cpu().numpy()]
    detected_labels = all_images_labels[0]

    for b, bb in enumerate(boxes_resized):
        if(detected_labels[b] != 0):
            rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=3,
                                    edgecolor='b', facecolor="none")
            voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor')

            score = all_images_scores[0][b]

            voc_label_to_id_map = {k: v + 1 for v, k in enumerate(voc_labels)}
            voc_id_to_label_map = {v: k for k, v in voc_label_to_id_map.items()}
            lbl_string = voc_id_to_label_map[int(detected_labels[b])] + "  " + f"{score:.2f}"
            axes.text(bb[0] + 2, (bb[1]) + 5, lbl_string, verticalalignment='center', color='white', fontsize=10, weight='bold',
                      bbox=dict(facecolor='blue', edgecolor='none'))
            axes.add_patch(rect)

    return fig

def save_detection(fig, output_folder):
    fig.savefig(output_folder)

def detect_objects(ai8x_model, image_path, output_folder):

    input_data = preprocess(image_path)

    model = AI8XInference(model_class="ai87fpndetector",
                        model_module="ai87-fpndetector",
                        model_checkpoint_path=Path(ai8x_model))
    
    # Generate inference
    locs, scores = model.generate_expected_output(input_data)

    locs = (locs) / 128.
    scores = scores / 2.**14

    # Isolate detected objects using non-maximum suppression
    all_locs, all_labels, all_scores = non_maximum_suppression(locs, scores)

    fig = display_detection(input_data, all_locs, all_labels, all_scores)
    save_detection(fig, output_folder)
    
    print(f"Detection complete. Output saved to {output_folder}")

def main(argv):
    """
    Basic FeaturePyramidNet ai8x Example
    Usage: python example.py <image file> [<image file> ...]
    """

    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    ai8x_path = os.path.join(root_path, "data", "model", "ai87-pascalvoc-fpndetector-qat8-q.pth.tar")
    imgs = argv or [os.path.join(root_path, "data", "input", "cat.png")]

    for img_file in imgs:

        base_name = Path(img_file).stem + ".jpg"
        output_name = os.path.join(output_folder, "detected_" + base_name)
        detect_objects(ai8x_path, img_file, output_name)

if __name__ == "__main__":
    main(sys.argv[1:])
