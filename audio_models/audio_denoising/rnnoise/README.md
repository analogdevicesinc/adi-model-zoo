# RNNoise

## Description

RNNoise is a noise reduction network that helps to remove noise from audio signals while maintaining any speech through the use of recurrent neural networks. This is a TFLite quantized version that takes traditional signal processing features and outputs gain values that can be used to remove noise from audio.

Trained on Noisy Speech database. Dataset Link: https://datashare.ed.ac.uk/handle/10283/2791

## License
Model: [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)
Dataset: [Creative Commons Attribution 4.0 International](https://spdx.org/licenses/CC-BY-4.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  Size (Bytes)       | 110 KB (113,472 bytes) |
|  Paper              | https://arxiv.org/pdf/1709.08243.pdf |

## Performance
| Platform  | Supported  |
|----------|------------|
| MAX78002 | ❌ |
| MAX32690 | ❌ |
| ADSP-SC835 | ❌ |

## Accuracy
| Metric | Value |
|--------|-------|
| Average PESQ | 2.945 |

**Note**: Perceptual Evaluation of Speech Quality (PESQ) is a standardized method for assessing the quality of speech signals. Scores typically range from 1.0 to 4.5, with higher scores indicating better quality.

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |

## Network Inputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| main_input_int8 | (1, 1, 42) | Pre-processed signal features extracted from 480 values of a 48KHz wav file |
| vad_gru_prev_state_int8 | (1, 24) | Previous GRU state for the voice activity detection GRU |
| noise_gru_prev_state_int8 | (1, 48) | Previous GRU state for the noise GRU |
| denoise_gru_prev_state_int8 | (1, 96) | Previous GRU state for the denoise GRU |

## Network Outputs

| Name  | Shape  | Description  |
|------|--------|--------------|
| Identity_int8 | (1, 1, 96) | Next GRU state for the denoise GRU |
| Identity_1_int8 | (1, 1, 22) | Gain values that can be used to remove noise from this audio sample |
| Identity_2_int8 | (1, 1, 48) | Next GRU state for the noise GRU |
| Identity_3_int8 | (1, 1, 24) | Next GRU state for the voice activity detection GRU |
| Identity_4_int8 | (1, 1, 1) | Probability that this audio sample contains voice activity |

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
