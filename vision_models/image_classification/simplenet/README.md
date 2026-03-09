# SimpleNet

## Description
SimpleNet performs image classification using a straightforward stack of convolutional blocks without residuals or depthwise separable layers. Spatial resolution is reduced using strided convolutions or max pooling, while channel depth increases across stages. After feature extraction, global average pooling compresses the feature maps, and a final fully connected layer produces the class logits followed by a Softmax layer.

Trained on the CIFAR100 dataset. Dataset Link: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

## License
Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)
Dataset: [MIT](https://spdx.org/licenses/MIT.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Size               | 8.40 MB (8,814,379 bytes) |
|  Paper              | [Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures](https://arxiv.org/abs/1608.06037) |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
| Metric                 | Value    |
|------------------------|----------|
| Top-1 Accuracy |0.6030 |
| Top-5 Accuracy |0.8623 |

**Note**: Top-K Accuracy measures if the correct label is among the top K predicted probabilities, common in multi-class problems with many categories. Higher values indicate better classification performance.

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | Q7 Fixed Point Format / 8-bit QAT |
| Architecture | SimpleNet |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| input | (3, 32, 32) | input image in CHW format |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| output | (1, 100) | confidence scores for CIFAR100 classes |

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
