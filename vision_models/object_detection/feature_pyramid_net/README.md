# Feature Pyramid Net

## Description

A Feature Pyramid Network (FPN) is a neural‑network architecture designed to improve multi‑scale object detection by combining the naturally occurring low‑resolution, semantically strong features from deep CNN layers with high‑resolution, semantically weak features from earlier layers through a top‑down pathway and lateral connections, producing a pyramid of feature maps that are both high in semantic meaning and high in spatial resolution across all scales.

Trained on the PascalVOC dataset. Dataset Link: [https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/)

## License
Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)
Dataset: [Creative Commons Attribution 2.5 Generic](https://spdx.org/licenses/CC-BY-2.5.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | PyTorch |
|  Size               | 25.16 MB (26,380,367 bytes) |
|  Paper              | [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) |

## Performance
| Platform | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
| Metric                 | Value    |
|------------------------|----------|
| Mean Average Precision |[0.50512] |

**Note**: Mean Average Precision (mAP) is a standard evaluation metric used in object detection to measure how accurately a model identifies and localizes objects in images. Scores typically range from 0.0 to 1.0, where higher values indicate better detection performance.

## Optimizations
| Optimization |  Value                            |
|--------------|-----------------------------------|
| Quantization | Q7 Fixed Point Format / 8-bit QAT |
| Architecture | Feature Pyramid Network           |

## Network Inputs
| Name  | Shape      | Description            |
|-------|------------|------------------------|
| input | (3, 256, 320) | input image dimensions in CHW format |

## Network Outputs
| Name           | Shape          | Description                                            |
|----------------|----------------|--------------------------------------------------------|
| regression     | (1, 10200, 4)  | location points of bounding boxes for detected objects |
| classification | (1, 10200, 21) | confidence scores for PascalVOC classes                |

## Inference

To run inference using the ai8x model, follow these steps:

1. Install the required libraries in your local environment with Python >= 3.10 using:

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
  