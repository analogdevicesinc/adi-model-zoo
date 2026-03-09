# Copyright © 2025-2026 Analog Devices, Inc. All Rights Reserved. This software is proprietary and confidential to Analog Devices, Inc. and its licensors.

import os
import sys
from os import listdir
import ai_edge_litert.interpreter as litert
import numpy as np
from PIL import Image as PImage

def preprocess_single(image_path):
    img = PImage.open(image_path).convert('L')
    img = img.resize((50, 50))
    return img

def preprocess_directory(image_path):
    filenames = sorted(listdir(image_path))
    loaded_images = []
    for image in filenames:
        path = os.path.join(image_path, image)
        loaded_images.append(preprocess_single(path))
    return loaded_images

def load_expected_outputs(path):
    with open(path) as f:
        return [line.rstrip() for line in f.readlines()]

def load_model(model_path):
    interpreter = litert.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def run_inference(interpreter, input_details, output_details, img):
    input_data = np.asarray(img, dtype=np.int8)
    input_data = np.resize(input_data, (1, 50, 50, 1))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return "Person" if output_data[0][0] < output_data[0][1] else "Not Person"

def evaluate_model(model_path, inputs_dir, gt_file):
    interpreter, input_details, output_details = load_model(model_path)
    inputs  = preprocess_directory(inputs_dir)
    outputs = load_expected_outputs(gt_file)

    num = min(len(inputs), len(outputs))
    num_correct = 0

    for idx in range(num):
        prediction = run_inference(interpreter, input_details, output_details, inputs[idx])
        expected   = outputs[idx]
        if prediction == expected:
            num_correct += 1
        print(f"Predicted: {prediction:<12} Expected: {expected}")

    acc = num_correct / num * 100
    print(f"\nAccuracy: {acc:.2f}% ({num_correct}/{num})")

def main(argv):
    root_path  = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_path, "data", "model", "vww2_50_50_INT8.tflite")

    if "--evaluate" in argv:
        inputs_dir = os.path.join(root_path, "data", "inputs")
        gt_file    = os.path.join(root_path, "data", "outputs", "output.txt")
        evaluate_model(model_path, inputs_dir, gt_file)
        return

    inputs_dir = os.path.join(root_path, "data", "inputs")
    imgs = [a for a in argv if not a.startswith("--")] or [
        os.path.join(inputs_dir, f) for f in sorted(listdir(inputs_dir))
    ]

    gt_file  = os.path.join(root_path, "data", "outputs", "output.txt")
    outputs  = load_expected_outputs(gt_file) if os.path.exists(gt_file) else None

    interpreter, input_details, output_details = load_model(model_path)

    num = min(len(imgs), len(outputs)) if outputs else len(imgs)
    num_correct = 0

    for idx in range(num):
        img        = preprocess_single(imgs[idx])
        prediction = run_inference(interpreter, input_details, output_details, img)
        if outputs:
            expected = outputs[idx]
            if prediction == expected:
                num_correct += 1
            print(f"Predicted: {prediction:<12} Expected: {expected}")
        else:
            print(f"Predicted: {prediction}")

    if outputs:
        acc = num_correct / num * 100
        print(f"\nAccuracy: {acc:.2f}% ({num_correct}/{num})")

if __name__ == "__main__":
    main(sys.argv[1:])
