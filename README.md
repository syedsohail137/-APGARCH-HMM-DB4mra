# High-Frequency Multi-Resolution Trading Pipeline

A zero-dependency, standalone modern C++ translation of a private institutional forecasting engine originally implemented across 6 linked `Excel 2010 (*.xlsm)` VBA modules. This engine couples statistical economics with discrete signal processing to generate 1-step ahead execution targets.

## ⚙️ Mathematical Engine Architecture

1. **Baum-Welch APGARCH-HMM Engine:** Solves a 2-State Hidden Markov Model using Forward/Backward scaling passes. Incorporates an Asymmetric Power GARCH framework running against a non-Gaussian Skewed Student-t distribution profile via a 9-coefficient Lanczos engine.
2. **Discrete Wavelet Denoising Pass:** Decomposes tracking errors across a 6-scale Multi-Resolution Pyramid via precise Daubechies-4 coefficients, thresholding intermediate components via adaptive non-linear Garrote Shrinkage Rules.
3. **Bedrosian Cross-Spectral Check:** Runs a native Discrete Fourier Transform (DFT) signature analyzer evaluating low-frequency trend properties relative to cycle anomalies. Validates separation laws to shield down-stream phase estimations from spectral bleeding.
4. **Hilbert Phasor Fusion Loop:** Convolutes extracted components via a 7-tap Hilbert window to establish analytical complex orthogonal dimensions, translating geometric offsets into definitive trading actions.
5. **Konishi Liquidity-Lag Optimizer:** Implements Hizuru Konishi's liquidity-lag distribution protocols, adjusting volume matrix weights selectively over standard market U-shaped profiling intervals.

## 🛠️ Build and Compilation Instructions

This workspace contains zero outside package dependencies. It compiles cleanly on any system via basic C++11 standard configurations or higher.

### Compile using GCC (Linux / macOS Terminal)
```bash
g++ -O3 -std=c++11 main.cpp -o TradingPipeline
```

### Run Execution Commands
Make sure you have an input data feed named `input_vwap.csv` containing your 256 data points before running:
```bash
./TradingPipeline
```

## 📊 File Layout Configuration
* `input_vwap.csv`: Input target containing a single column array of exactly 256 sequential VWAP data entries.
* `output_predictions.csv`: Exporter target tracking synchronized parameters across all transformed processing columns.
