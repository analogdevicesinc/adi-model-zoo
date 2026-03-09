# GenreNet (Conv2D)

## Description
GenreNet is a Deep-based approach for music genre classification.

## License
Model: [MIT](https://spdx.org/licenses/MIT.html/)

Dataset: [MIT](https://spdx.org/licenses/MIT.html/)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | PyTorch |
|  SHA-1 Hash         | n/a |
|  Size (Bytes)       | 15671094 |
|  Provenance         | https://link.springer.com/article/10.1007/s42979-024-03493-x |
|  Paper              | https://link.springer.com/article/10.1007/s42979-024-03493-x |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ❌ |
| ADSP-SC835 | ✅ |

## Accuracy
Dataset:  GTZAN Genre Collection

| Metric  | With STFT and Magphase |
|--------|-------|
| Accuracy  | 0.8450 |
| Precision | 0.9824 |
| Recall | 0.9635 |
| RMSerror | 0.9729 |
| F1 score  | 0.1573 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | int8 |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| main_input | (1, 1, 128, 128) | Pre-processed signal features extracted from 480 values of a 48KHz wav file |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (1, 10) | Probability States for all 10 labels |

## Inference

To run inference using the tflite model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -U setuptools
    pip install -r requirements.txt
    ```

2. Run the inference code with the path to a wav file as input using one of the tflite models:

    ```shell
    python example.py
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
  All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
