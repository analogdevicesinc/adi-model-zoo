# Copyright © 2026 Analog Devices, Inc. All Rights Reserved. This software is proprietary and confidential to Analog Devices, Inc. and its licensors.

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference

def preprocess(image_path, input_size=112):
    transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875), antialias=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    tensor = tensor * 128.0

    return tensor.detach().numpy()

def classify(ai8x_model, image_path, labels):
    input_data = preprocess(image_path, input_size=112)

    model = AI8XInference(
        model_class="ai87imageneteffnetv2",
        model_module="ai87net_imagenet_effnetv2",
        model_checkpoint_path=Path(ai8x_model),
        hardware="MAX78002"
    )

    output = model.generate_expected_output(input_data)
    output = output.detach().numpy().flatten()
    predicted_idx = int(np.argmax(output))
    predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else "UNKNOWN"

    return predicted_idx, predicted_label

def write_classifications(img_path, predicted_idx, predicted_label, output_folder):
    base_name = Path(img_path).stem + ".txt"
    output_path = os.path.join(output_folder, "classified_" + base_name)

    lines = [
        "EFFNETV2".center(44) + "\n",
        f"INFERENCE FOR: {Path(img_path).name}".center(44) + "\n",
        "=" * 44 + "\n",
        f"{'Image':<20} : {Path(img_path).name}\n",
        f"{'Predicted Index':<20} : {predicted_idx}\n",
        f"{'Predicted Label':<20} : {predicted_label}\n",
        "-" * 44 + "\n",
    ]

    with open(output_path, 'w') as file:
        for line in lines:
            file.write(line)

    print("EFFNETV2".center(44))
    print(f"INFERENCE FOR: {Path(img_path).name}".center(44))
    print("=" * 44)
    print(f"{'Image':<20} : {Path(img_path).name}")
    print(f"{'Predicted Index':<20} : {predicted_idx}")
    print(f"{'Predicted Label':<20} : {predicted_label}")
    print("-" * 44)
    print(f"Classification complete. Output saved to {output_path}")

def main(argv):

    root_path = os.path.dirname(__file__)
    model_path = os.path.join(root_path, "data", "model", "ai87-imagenet-effnet2-q.pth.tar")
    labels_path = os.path.join(root_path, "data", "model", "imagenet_classes.txt")
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    imgs = argv or [os.path.join(root_path, "data", "input", "banana.jpg")]

    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    for img in imgs:
        predicted_idx, predicted_label = classify(model_path, img, labels)
        write_classifications(img, predicted_idx, predicted_label, output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])
