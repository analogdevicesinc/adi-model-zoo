# Data Processing for Motor Fault Detection

The motor fault detection task expects **motor vibration data** in the X, Y, Z directions.

To collect data for training, ADI's **ADXL356 Tri-Axis MEMS Accelerometer** was used to record data for 2 seconds in 20 kHz sampling frequency. The original dataset can be found here: [Motor Fault Sample Dataset](https://github.com/analogdevicesinc/CbM-Datasets/tree/main)

## CSV Input Format

The model's data processor (used in [example.py](example.py) and in [EdgeBench](https://edgebench.app.analog.com/)) accepts `.CSV` files in the following format:

```
time (in seconds) ; voltage x ; voltage y ; voltage z ; initial x ; initial y ; initial z
```

- Each value must be separated by a semicolon (;)
- Only the first row requires the data for initial x, y, and z values. These initial reference values are used to compute per-axis acceleration.

A valid sample is shown below:

```
0      ; 0.88896352 ; 0.9082635  ; 0.8902815  ; 0.8906 ; 0.9076 ; 0.8894
5e-005 ; 0.8934024  ; 0.90861553 ; 0.88492393 
0.0001 ; 0.89479351 ; 0.90739429 ; 0.87852722 
...
```

## Model Input Format

When running the model directly, each input window must be provided as a **binary stream containing 192 uint32 values**. This binary representation corresponds to a quantized (256, 3) FFT, where 256 frequency bins are computed for each of the three accelerometer axes.

Raw CSV data can be converted into this format by [data_processor.py](data_processing/data_processor.py) through the following preprocessing pipeline:

1. **Acceleration conversion** — Raw voltage measurements are converted to acceleration in g-units: a = 50 × (voltage - initial_value) for each axis.
2. **Downsampling** — The signal is decimated from 20 kHz to 2 kHz.
3. **Windowing** — The signal is split into 0.25-second windows (500 samples each) with 75% overlap.
4. **FFT** — A 1D FFT is computed per axis per window. Only the first 256 frequency bins are kept, and the first 3 and last 10 bins are zeroed out.
5. **Normalization** — Each window is L2-normalized across axes, then min-max normalized per instance. Zeroed bins are set to 0.5.
6. **Quantization and packing** — The (256, 3) float array is quantized to int8 and packed into 192 uint32 values.

The data processor returns this as a numpy array of shape (192,) and of type uint32.

## Model Output Format

The autoencoder reconstructs the input, producing 192 uint32 values in the same packed format. The postprocess function in [data_processor.py](data_processing/data_processor.py) supports two output modes:

- **loss** — Unpacks the reconstructed bitstream to (256, 3) int8, then computes MSE loss against the original reference signal. Higher loss indicates the input deviates from the training distribution, suggesting a fault.
- **reconstructed** — Returns the unpacked (256, 3) int8 array directly for visualization or further analysis.
