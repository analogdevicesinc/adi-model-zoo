# MicroNet VWW-2 INT8

## Description
This is a fully quantized version (asymmetrical int8) of the MicroNet VWW-2 model developed by Arm, from the MicroNets paper. It is trained on the 'Visual Wake Words' dataset, more information can be found here: https://arxiv.org/pdf/1906.05721.pdf.

## License
Model: [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

Dataset: [Visual Wake Words](https://spdx.org/licenses/CC-BY-4.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 5d887ca438c0a7feeed3c8c22dce99b55565c8ea |
|  Size (Bytes)       | 280384 |
|  Provenance         | https://arxiv.org/pdf/2010.11267.pdf |
|  Paper              | https://arxiv.org/pdf/2010.11267.pdf |

| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ✅ |
| ADSP-SC835 | ❌ |

## Accuracy
Dataset: Visual Wake Words

| Metric | Value |
|--------|-------|
| Accuracy | 0.768 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |

## Network Inputs
<table>
    <tr>
        <th width="200">Input Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>input</td>
        <td>(1, 50, 50, 1)</td>
        <td>A 50x50 input image.</td> 
    </tr>
</table>

## Network Outputs
<table>
    <tr>
        <th width="200">Output Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>Identity</td>
        <td>(1, 2)</td>
        <td>Per-class confidence across the two classes (0=no person present, 1=person present).</td> 
    </tr>
</table>

## Inference

To run inference using the tflite model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -r requirements.txt
    ```

2. Evaluate the model via shell script:

    ```shell
    python example.py 
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
