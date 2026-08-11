# MicroNet for Anomaly Detection

## Description

MicroNet AD Small is a lightweight convolutional neural network designed for unsupervised anomaly detection on edge and TinyML devices. Based on the MicroNets architecture, it is trained as a self-supervised machine-ID classifier using only normal machine sounds. During inference, the negative log-likelihood (NLL) of the expected machine ID is used as the anomaly score to detect abnormal operating conditions. This model uses the MicroNet AD Small architecture for industrial sound anomaly detection.

Trained on the DCASE 2020 Challenge Task 2 dataset (slide rail / slider machine type), sourced from the MIMII industrial machine sound dataset. Dataset Link: [https://zenodo.org/records/3384388](https://zenodo.org/records/3384388)

## License

Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)
Dataset: [CC BY-SA 4.0](https://spdx.org/licenses/CC-BY-SA-4.0.html)

## Network Information

| Network Information |  Value         |
|---------------------|----------------|
|  Size               | 247 KB (253,632 bytes) |

## Performance

| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ✅ |
| ADSP-SC835 | ❌ |

## Accuracy

| Metric | Value |
|---------|-------:|
| AUC | 0.8768 |
| pAUC (p=0.1) | 0.5853 |

**Note**: Area Under the ROC Curve (AUC) and partial AUC (pAUC) are standard metrics for anomaly detection. Higher values indicate better separation between normal and anomalous machine sounds. 

## Optimizations

| Optimization |  Value  |
|--------------|---------|
| Quantization | int8 |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| input | (1, 32, 32, 1) | 64 steps of a log mel spectrogram resized to 32x32 |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (1, 4) | raw logits corresponding to different machine IDs |

## Inference

To run inference using the tflite model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -r requirements.txt
    ```

2. Run the inference code with the path to a csv file as input using the tflite model:

    ```shell
    python example.py ./data/input/sample_signal.csv
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:** 
  All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
