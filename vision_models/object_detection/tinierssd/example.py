# Copyright © 2026 Analog Devices, Inc. All Rights Reserved. This software is proprietary and confidential to Analog Devices, Inc. and its licensors.

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference
from data_processing import data_processor
import data_processing.utils.object_detection_utils as obj_detect_utils

def preprocess(image_path):
    return data_processor.preprocess(image_path, pack=False).astype(np.float32)

def non_max_suppression(locs, scores, **kwargs):
    num_classes = 2
    min_score = kwargs.get('min_score', 0.4)
    max_overlap = kwargs.get('max_overlap', 0.1)
    top_k = kwargs.get('top_k', 20)

    prior_boxes = obj_detect_utils.create_prior_boxes()
    all_images_boxes, all_images_labels, all_images_scores = obj_detect_utils.detect_objects(
        locs, scores, prior_boxes, min_score=min_score, max_overlap=max_overlap, top_k=top_k
    )

    return all_images_boxes, all_images_labels, all_images_scores

def display_detections(image, all_images_boxes, all_images_labels, all_images_scores):
    img_to_plot = (image + 128).astype(np.uint8).transpose([1, 2, 0])  # (C,H,W) -> (H,W,C)

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes.imshow(img_to_plot)
    axes.set_title("QR Detection", fontsize=14)

    voc_labels = {0: "QR Code"}
    voc_label_to_id_map = {k: v + 1 for v, k in enumerate(voc_labels.values())}
    voc_id_to_label_map = {v: k for k, v in voc_label_to_id_map.items()}

    boxes_resized = [
        [box[0] * img_to_plot.shape[1], box[1] * img_to_plot.shape[0],
         box[2] * img_to_plot.shape[1], box[3] * img_to_plot.shape[0]]
        for box in all_images_boxes[0].detach().cpu().numpy()
    ]
    detected_labels = all_images_labels[0]

    for b, bb in enumerate(boxes_resized):
        rect = patches.Rectangle(
            (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
            linewidth=3, edgecolor='b', facecolor="none"
        )
        lbl_string = voc_id_to_label_map[int(detected_labels[b])]
        score = all_images_scores[0][b].item()
        axes.text(bb[0], bb[1] + 4, f"{lbl_string} {score:.2f}",
                  color='white', fontsize=11, weight='bold',
                  verticalalignment='bottom',
                  bbox=dict(facecolor='blue', edgecolor='none', pad=2))
        axes.add_patch(rect)

    return fig

def detect_objects(ai8x_model, image_path):
    input_data = preprocess(image_path)

    model = AI8XInference(
        model_class="ai85tinierssdqr",
        model_module="ai85net_tinierssd_kpts_qr",
        model_checkpoint_path=Path(ai8x_model)
    )

    locs, scores = model.generate_expected_output(input_data)
    locs = locs/128
    scores = scores/128
    
    all_locs, all_labels, all_scores = non_max_suppression(locs, scores)

    fig = display_detections(input_data, all_locs, all_labels, all_scores)
    
    return fig

def main(argv):
    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)

    ai8x_path = os.path.join(root_path, "data", "model", "ai85-qrcode-tinierssd-kpts-qat8-q.pth.tar")
    imgs = argv or [os.path.join(root_path, "data", "input", "sample_input.jpg")]
    
    for img in imgs:
        base_name = Path(img).stem + ".jpg"
        output_name = os.path.join(output_folder, "detected_" + base_name)
        fig = detect_objects(ai8x_path, img)
        fig.savefig(output_name, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])
