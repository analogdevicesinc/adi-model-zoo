# TinierSSD

## Description 
A tinier Single Shot Detector (SSD) model for QR code detection, trained and optimized for deployment on the MAX78000 device.

## License
Model: [MIT](https://spdx.org/licenses/MIT.html)

Dataset: [MIT](https://spdx.org/licenses/MIT.html)

## Network Information
| Parameter  | Value  |
|------------|--------|
| Framework  | PyTorch |
| Size       | 4462000 Bytes  |
| Paper      | https://arxiv.org/abs/1512.02325 |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ✅ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
| Metric  | Value  |
|--------|--------|
| mAP | 0.89960 |

## Optimizations
| Optimization  | Value  |
|--------------|--------|
| Quantization | INT8 |

## Network Inputs
<table>
    <tr>
        <th width="200">Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>input</td>
        <td>(1, 3, 120, 160)</td>
        <td>RGB input image resized to 120×160 (H×W). Channels-first format.</td> 
    </tr>
</table>

## Network Outputs
<table>
    <tr>
        <th width="200">Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>loc</td>
        <td>(1, 16, 120, 160)</td>
        <td>Bounding box localization predictions</td> 
    </tr>
    <tr>
        <td>cl</td>
        <td>(1, 8, 120, 160)</td>
        <td>Classification scores</td> 
    </tr>
</table>

## Inference

To run inference using the ai8x model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -U setuptools
    pip install -r requirements.txt
    ```

2. Run the inference code with the path to a image file as input using one of the ai8x models:

    ```shell
    python example.py
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
  All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
  