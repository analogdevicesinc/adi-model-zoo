# Anomaly Detection Autoencoder INT8

## Description
Autoencoder network designed for anomaly detection. 

## License
Model: [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html)

Dataset: [MIMII](https://spdx.org/licenses/CC-BY-SA-4.0.html)

## Dataset

Trained under the MIMII Dataset for fan, pump, slide rail, and valve. Has [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html) License

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | n/a |
|  Size (Bytes)       | 307280 |
|  Provenance         | https://arxiv.org/pdf/2006.10417 |
|  Paper              | https://arxiv.org/pdf/2006.10417 |

| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ✅ |
| ADSP-SC835 | ❌ |

## Accuracy
Dataset: MIMII Dataset - Fan 

| Machine Type | AUC | pAUC |
|--------|--------|-------|
| fan | 0.51633 | 0.52276 |
| pump | 0.51174 | 0.50701 |
| slider | 0.45907 | 0.48953 |
| valve | 0.49192 | 0.49391 |

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
        <td>(n, 196, 640)</td>
        <td>a Mel Spectrogram of the waveform with resolution 196 x 640. n is the number of samples.</td> 
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
        <td>output signal</td>
        <td>(n, 196, 640)</td>
        <td>Output Mel Spectrogram of the waveform with resolution 196 x 640. n is the number of samples.</td> 
    </tr>
</table>

## Inference

To run inference using the tflite model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -r requirements.txt
    ```

2. Evaluate the model via shell script

    ```shell
    python example.py --model model/model_fan_quant.tflite --input data/input/fan 
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
