# Dual-signal Transformation LSTM Network

## Description

Dual-signal Transformation LSTM Network (DTLN) combines a short-time Fourier transform (STFT) and a learned analysis and synthesis basis in a stacked-network approach with less than one million parameters. The network is capable of real-time processing (one frame in, one frame out) and reaches competitive results. Combining these two types of signal transformations enables the DTLN to robustly extract information from magnitude spectra and incorporate phase information from the learned feature basis.

Trained on 500 hours of noisy speech from the DNS challenge dataset. Dataset Link: [https://github.com/microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)

## License
Model: [MIT](https://spdx.org/licenses/MIT.html)
Dataset: [Creative Commons Attribution 4.0 International](https://spdx.org/licenses/CC-BY-4.0.html)

## Network Information
| Parameter  | Value  |
|------------|--------|
| Framework  | TensorFlow Lite |
| Size       | 1.39 MB (1,459,376 bytes) + 2.39 MB (2,510,296 bytes) |
| Paper      | https://www.isca-speech.org/archive/interspeech_2020/westhausen20_interspeech.html |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ❌ |
| ADSP-SC835 | ✅ |

## Accuracy
| Metric  | Value  |
|--------|--------|
| PESQ | 2.95 |

**Note**: Perceptual Evaluation of Speech Quality (PESQ) is a standardized method for assessing the quality of speech signals. Scores typically range from 1.0 to 4.5, with higher scores indicating better quality.

## Optimizations
| Optimization  | Value  |
|--------------|--------|
| Precision | FP32 |
| Quantization | INT16 |

## Network Inputs

Network #1
| Name  | Shape  | Description  |
|------|--------|--------------|
| in_mag | (1, 1, 257) | Preprocessed magnitude chunk of input audio |
| states_1 | (1, 2, 128, 2) | LSTM state |

Network #2
| Name  | Shape  | Description  |
|------|--------|--------------|
| estimated_block | (1, 1, 512) | Time domain block derived from output of Network #1 |
| states_2 | (1, 2, 128, 2) | LSTM state |

## Network Outputs
Network #1
| Name  | Shape  | Description  |
|------|--------|--------------|
| out_mask | (1, 1, 257) | Time-frequency mask output of Network #1 |
| states_1 | (1, 2, 128, 2) | LSTM state |

Network #2
| Name  | Shape  | Description  |
|------|--------|--------------|
| out_block | (1, 1, 512) | Denoised output of Network #2 |
| states_2 | (1, 2, 128, 2) | LSTM state |

## Inference

To run inference using the tflite model, follow these steps:

1. Install the required libraries in your local environment with Python 3.x using:

    ```shell
    pip install -U setuptools
    pip install -r requirements.txt
    ```

2. Run the inference code with the path to an audio file as input using one of the tflite models:

    ```shell
    python example.py ./data/input/test.wav
    ```

## Requirements  
### Software  
- **Python Version:** >=3.10  
- **Dependencies:**  
  All required Python packages are specified in the [`requirements.txt`](requirements.txt) file.
