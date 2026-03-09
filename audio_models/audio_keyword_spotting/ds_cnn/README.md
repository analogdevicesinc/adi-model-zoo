# Depthwise-Separable CNN for Keyword Spotting

## Description
This is a Depthwise Separable Convolutional Neural Network (DS-CNN) Small model developed by Arm, from the Hello Edge paper. This uses Google Speech Commands V2 dataset which contains a set of one-second .wav audio files, each containing a small set of commands.

## License
Model: [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

Dataset: [Google Speech Commands V2](https://spdx.org/licenses/CC-BY-4.0.html)

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  Datatype           | int8 |
|  SHA-1 Hash         | 504f8e7bfa5c0f15c5475e5d08637b3b8aad0972 |
|  Size (Bytes)       | 503816 |
|  Provenance         | https://arxiv.org/abs/1711.07128 |
|  Training           | Trained by Arm |
|  Paper | https://arxiv.org/abs/1711.07128 |

## DataSet
| Dataset Information | Value |
|--------|-------|
| Name | Google Speech Commands test set |

## Accuracy

| Metric | Value |
|--------|-------|
| Accuracy | 94.52% |

## HW Support
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ✅ |
| ADSP-SC835 | ✅ |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |

## Network Inputs
| Input Node Name | Shape | Example Use Case |
|-----------------|-------|-----------------|
| input | (1, 490) | The input is a processed MFCCs |

## Network Outputs
| Output Node Name | Shape | Example Use Case |
|-----------------|-------|-----------------|
| Identity | (1, 12) | The probability on 12 keywords |

## Inference

To run inference using the tflite model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -U setuptools
    pip install -r requirements.txt
    ```
2. Run the inference:

    ```shell
    python example.py
    ```
## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
  All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
