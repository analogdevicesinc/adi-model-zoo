# AudioNet (Conv1D)

## Description
Conv1D AudioNet is a CNN-based model for keyword spotting.

## License
Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html/)

Dataset: [Google Speech Commands V2](https://spdx.org/licenses/CC-BY-4.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | PyTorch |
|  SHA-1 Hash         | n/a |
|  Size (Bytes)       | n/a |
|  Provenance         | https://arxiv.org/abs/1711.07128 |
|  Training           | Trained by Arm |
|  Paper | https://arxiv.org/abs/1711.07128 |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
Dataset:  KWS20
| Metric  | Macro  | Weighted |
|----------|------------|------------|
| Accuracy| 0.8634 | 0.8526 |
| Precision | 0.8361 | 0.8650 |
| Recall | 0.8979 | 0.8526 |
| F1 Score | 0.8634 | 0.8510 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | Q7 Fixed Point Format |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| input | (128, 128) | Time and Feature dimensions |
## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (1, 21) | number of target keywords + 1 unknown |

## Inference

To run inference using the PyTorch model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -r requirements.txt
    ```

2. Evaluate the model via shell script

    ```shell
    python example.py
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**
