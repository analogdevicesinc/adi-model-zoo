# UNet

## Description
U‑Net is an encoder–decoder CNN for image segmentation that combines a contracting path for contextual feature extraction with an expanding path for precise localization, using skip connections to merge high‑resolution shallow features with deep semantic features, ultimately producing pixel‑wise class labels across the image.

Trained on the AISegment dataset. Dataset Link: [https://www.kaggle.com/datasets/laurentmih/AISegmentcom-matting-human-datasets](https://www.kaggle.com/datasets/laurentmih/AISegmentcom-matting-human-datasets)

## License
Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)
Dataset: [MIT](https://spdx.org/licenses/MIT.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | PyTorch        |
|  Size               | 3.25 MB (3,411,334 bytes) |
|  Paper              | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
| Metric                 | Value    |
|------------------------|----------|
| Top-1 Accuracy         | 0.98452  |

**Note**: Top‑1 accuracy represents the proportion of pixels for which the model’s highest‑scoring predicted class matches the true class. This serves as a basic measure of overall pixel‑wise correctness in segmentation. Higher values indicate better segmentation performance.

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | Q7 Fixed Point Format / 8-bit QAT |
| Architecture | UNet          |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| input | (48, 48, 48) | folded input image in CHW format |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (1, 32, 48, 48) | segmented output mask |

## Inference

To run inference using the ai8x model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -r requirements.txt
    ```

2. Run the inference code with the path to an image file as input using the ai8x model:

    ```shell
    python example.py ./data/input/sample_input.jpg
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:** 
 All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
