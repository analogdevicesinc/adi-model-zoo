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

sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))

from third_party.ai8x.util_ai8x_inference import AI8XInference
from third_party.ai8x.util_ai8x_inference import ai8x_softmax_2

def preprocess(image_path):

    return data_processor.preprocess(image_path, pack=False).astype(np.float32)

def get_class_predictions(ai8x_model, input_data):

    model = AI8XInference(model_class="ai87netmobilenetv2cifar100_m0_75",
                    model_module="ai87net-mobilenet-v2",
                    model_checkpoint_path=Path(ai8x_model))
    
    # Get class inference
    id = model.generate_expected_output(input_data)
    id = id.detach().numpy()

    # Apply softmax to the output inference
    class_preds = ai8x_softmax_2(id[0])

    return class_preds

def write_classifications(class_preds, output_folder):

    classes = np.array(['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
                        'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
                        'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
                        'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine',
                        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
                        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                        'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank',
                        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
                        'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'])

    # Sort scores and labels
    sort_idx = np.argsort(-class_preds)
    labels_sorted = classes[sort_idx]
    scores_sorted = class_preds[sort_idx]

    base_name = output_folder.split("\\")[-1].replace("classified_", "")

    lines = ["MOBILENETV2".center(44)+"\n", f'INFERENCE FOR: {base_name}'.center(44)+"\n", "="*44 +"\n"]

    # Format predictions based on labels and scores
    for i in range(len(class_preds)):
        lines.append(f"{i+1:>4}. {labels_sorted[i]:>13} : {scores_sorted[i]:.20f} \n")

    # Write lines to folder
    with open(output_folder, 'w') as file:
        for line in lines:
            file.write(line)
    file.close()

def classify_image(ai8x_model, image_path, output_folder):

    input_data = preprocess(image_path)
    class_scores = get_class_predictions(ai8x_model, input_data)
    write_classifications(class_scores, output_folder)
    print(f"Classification complete. Output saved to {output_folder}")

def main(argv):
    """
    Basic MobileNetv2 ai8x Example
    Usage: python example.py <image file> [<image file> ...]
    """

    root_path = os.path.dirname(__file__)
    output_folder = os.path.join(root_path, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    ai8x_path = os.path.join(root_path, "data", "model", "ai87-cifar100-mobilenet-v2-0.75-qat8-q.pth.tar")
    imgs = argv or [os.path.join(root_path, "data", "input", "cup.png")]

    for img_file in imgs:

        base_name = Path(img_file).stem + ".txt"
        output_name = os.path.join(output_folder, "classified_" + base_name)
        classify_image(ai8x_path, img_file, output_name)

if __name__ == "__main__":
    main(sys.argv[1:])
