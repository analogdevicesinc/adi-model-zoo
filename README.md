# ADI Model Zoo

## Introduction

This repository provides a collection of example machine learning models designed to support rapid evaluation, benchmarking, and prototyping on Analog Devices hardware and embedded platforms. It enables developers to explore a variety of machine learning workloads and accelerate the development of intelligent edge applications.

The ADI Model Zoo includes ready‑to‑use **pretrained models** for **computer vision**, **audio analytics**, and **anomaly detection**. Each model is packaged with pretrained weights, sample data, preprocessing utilities, and example inference scripts to simplify experimentation and integration into end‑to‑end workflows.

## Structure

The repository is structured as follows:

```
adi-model-zoo/
├──── audio_models/
│       ├──── audio_denoising/
│       └──── audio_genre_identification/
├──── sensor_models/
│       ├──── anomaly_detection/
│       └──── motor_fault_detection/
├──── vision_models/
│       ├──── image_classification/
│       ├──── image_segmentation/
│       ├──── object_detection/
│       └──── visual_wake_word/
└──── README.md
```

Models are organized according to their primary application domain:

* [Vision Models](#vision-models) – models for image classification, segmentation, detection, and other visual tasks
* [Audio Models](#audio-models) – models for audio processing, denoising, and classification
* [Sensor Models](#sensor-models) – models for time‑series and sensor‑driven analytics

For details on supported file formats and instructions for running the example scripts, refer to the [Usage](#usage) section.

### Individual Model Structure

Each model has the following internal structure:

```
model/
├──── data/
│       ├──── input/        # Input data
│       ├──── model/        # Model files
│       └──── output/       # Expected/reference outputs
├──── data_processing/      # Optional preprocessing/postprocessing scripts
├──── definition.yaml       # Model metadata
├──── example.py            # Example inference script
├──── README.md             # Model-specific documentation
└──── requirements.txt      # Dependencies
```

This structure ensures that all models are self‑contained, easy to run, and consistent across domains.

## Models

### Vision Models

This collection of models takes images (`.JPG` or `.PNG` file format) as input.

* **Image Classification** - Assigns a label to an entire image
* **Image Segmentation** -  Produces pixel‑level classifications to separate objects/regions within an image
* **Object Detection** - Identifies and localizes objects in an image by drawing bounding boxes and classifying each detected region
* **Visual Wake Word** - Detects the presence or absence of a specific trigger object for low‑power devices

| Model             | Task                  | Metrics         |
|-------------------|-----------------------|-----------------|
| EfficientNet      | Image Classification  | Top-1: 0.6197   |
| MobileNetV2       | Image Classification  | Top-1: 0.6471   |
| SimpleNet         | Image Classification  | Top-1: 0.6030   |
| UNet              | Image Segmentation    | Top-1: 0.9845   |
| FeaturePyramidNet | Object Detection      | mAP: 0.50512    |
| TinierSSD         | Object Detection      | mAP: 0.89960    |
| Micronet          | Visual Wake Word      | Accuracy: 0.768 |

### Audio Models

This collection of models takes audio (`.MP3` or `.WAV` file format) as input.

* **Audio Denoising** - Removes interference from audio signals while preserving speech quality
* **Genre Identification** -  Classifies audio clips into genres based on acoustic features
* **Key Word Spotting** - Detects/classifies short spoken trigger phrases in streaming audio

| Model             | Task                  | Metrics          |
|-------------------|-----------------------|------------------|
| DTLN              | Audio Denoising       | PESQ: 2.950      |
| RNNoise           | Audio Denoising       | PESQ: 2.945      |
| GenreNet Conv2D   | Genre Identification  | Accuracy: 0.8450 |
| Conv1D AudioNet   | Key Word Spotting     | Accuracy: 0.8634 |
| DS CNN            | Key Word Spotting     | Accuracy: 0.9452 |

### Sensor Models

This collection of models takes time-series data (`.CSV` file format) as input.

* **Motor Fault Detection** - Identifies abnormal patterns in motor signals to detect mechanical faults or degradation
* **Anomaly Detection** - Flags unusual behavior in generic sensor data streams

| Model             | Task                  | Metrics                     |
|-------------------|-----------------------|-----------------------------|
| Autoencoder       | Motor Fault Detection | MSE: 0.02205                |
| Autoencoder       | Anomaly Detection     | AUC: 0.51633, pAUC: 0.52276 |

## Usage

Each model in the ADI Model Zoo comes with sample data, data processing utilities, and an example inference script. This section outlines how to set up the environment and run the models.

### File Requirements

For specific model categories, the following file formats are accepted:

| Category | File Format    |
|----------|----------------|
| Vision   | `.PNG`, `.JPG` |
| Audio    | `.MP3`, `.WAV` |
| Sensor   | `.CSV`         |

### Running the Example

1. Every model directory includes a `requirements.txt` file. Install the required libraries in your local environment using:

    ```shell
    pip install -U setuptools
    pip install -r requirements.txt
    ```
    Check the model's `README.md` file for the required Python version.

2. Run the inference code with the path to the input file using one of the models:

    ```shell
    python example.py data/input/input.jpg
    ```

    Make sure you are working in the **correct model folder** (the one containing example.py) before running.

### Using Your Own Data

When using custom data, you may specify a custom path when running the example script:

```shell
python example.py [path_to_file]
```
