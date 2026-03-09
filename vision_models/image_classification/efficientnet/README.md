# EfficientNet

## Description
EfficientNetv2 is used for Image Classification. Trained using ImageNet dataset.

## License
Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)

Dataset: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | PyTorch |
|  SHA-1 Hash         | n/a |
|  Size (Bytes)       | n/a |
|  Paper              | https://arxiv.org/abs/1905.11946 |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
Dataset:  ImageNet Dataset

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | Q7 Fixed Point Format |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| input | (112,112) | input dimensions |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (1, 1000) | output label |

## Inference

To run inference using the pytorch model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -U setuptools
    pip install -r requirements.txt
    ```
    
2. Run the inference code:

    ```shell
    python example.py
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
