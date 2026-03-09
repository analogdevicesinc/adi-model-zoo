# Autoencoder for Motor Fault Detection

## Description

Autoencoder is a type of artificial neural network used for unsupervised learning to learn efficient codings of unlabeled data. Consisting of an encoder and a decoder, the encoder compresses the input data into a lower-dimensional latent representation, while the decoder reconstructs the original data from this representation. This model uses the autoencoder architecture for motor fault detection.

Trained on the Motor Fault Sample dataset from Analog Devices, Inc. Dataset Link: [https://github.com/analogdevicesinc/CbM-Datasets/tree/main](https://github.com/analogdevicesinc/CbM-Datasets/tree/main)

## License

Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)
Dataset: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information

| Network Information |  Value         |
|---------------------|----------------|
|  Size               | 1.59 MB (1,669,366 bytes) |

## Performance

| Platform  | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy

| Metric                 | Value    |
|------------------------|----------|
| Mean Squared Error     | 0.02205  |

**Note**: Mean Squared Error (MSE) is the reconstruction error that rises when a signal deviates from normal behavior, with healthy data typically around 0.01–0.02 and faulty conditions often exceeding 0.03 or higher depending on severity.

## Optimizations

| Optimization |  Value  |
|--------------|---------|
| Quantization | Q7 Fixed Point Format / 8-bit QAT |
| Architecture | Autoencoder |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| input | (256, 3) | signal length, channels |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (256, 3) | decoded signal |

## Inference

To run inference using the ai8x model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -r requirements.txt
    ```

2. Run the inference code with the path to a csv file as input using the ai8x model:

    ```shell
    python example.py ./data/input/sample_signal.csv
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:** 
  All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
