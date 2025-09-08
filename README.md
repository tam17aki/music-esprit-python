# High-Resolution Parameter Estimation for Sinusoidal Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of high-resolution parameter estimation algorithms for sinusoidal signals including **MUSIC (Spectral/Root)** and **ESPRIT**. The project is structured with an object-oriented approach, emphasizing code clarity, reusability, and educational value.

This work is inspired by the foundational papers in subspace-based signal processing and aims to provide a practical and understandable guide to these powerful techniques.

## Features

- **High-Resolution Algorithms**: Implements spectral estimation techniques that surpass the resolution limits of the classical Fast Fourier Transform (FFT).
- **Multiple Methods Implemented**:
  - **Spectral MUSIC**: Frequency estimation via spectral peak-picking.
  - **Root MUSIC**: High-accuracy frequency estimation via polynomial rooting.
  - **Min-Norm**: A variant of MUSIC that can reduce computational cost by using a single, optimized vector from the noise subspace. Both spectral and root-based versions are implemented.
  - **ESPRIT**: A computationally efficient method that estimates frequencies directly without spectral search.
  - **HOYW**: A robust method based on the autocorrelation function and an AR model of the signal, enhanced with SVD-based rank truncation.
- **Full Parameter Estimation**: Not just frequencies, but also amplitudes and phases are estimated using a subsequent least-squares fit.
- **Enhanced Accuracy with Forward-Backward Averaging**: Improves estimation accuracy in low SNR or short data scenarios. This is implemented elegantly via a `ForwardBackwardMixin` class, showcasing a reusable and extensible design.
- **Object-Oriented Design**: Algorithms are encapsulated in clear, reusable classes (`SpectralMusicAnalyzer`, `RootMusicAnalyzer`, etc.), promoting clean code and extensibility.
- **Demonstration Script**: Includes a command-line interface (`main.py`) to easily run experiments and compare the performance of different algorithms.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/tam17aki/music-esprit-python.git
    cd music-esprit-python
    ```

2.  (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point is `main.py`, which allows you to run a comparative analysis of MUSIC and ESPRIT.

### Basic Example

To run a demonstration with default parameters (a signal with three sinusoids at 440, 460, and 480 Hz):

```bash
python main.py
```

#### Example Output
Running the command above will produce an output similar to this:

```
--- Experiment Setup ---
Sampling Frequency: 44100.0 Hz
Signal Duration:    100 ms
True Frequencies:   [440. 460. 480.] Hz
True Amplitudes:    [1.41102781 1.40052413 1.44984503]
True Phases:        [-3.04853895 -0.47321629  3.10837411] rad
SNR:                30.0 dB
Number of Grid Points:  8192

--- Running Spectral MUSIC ---

--- Estimation Results ---
Est Frequencies: [438.79257722 460.32840923 479.17226224] Hz
Est Amplitudes:  [1.4193478  1.40920727 1.36681428]
Est Phases:      [-2.67745079 -0.56606888 -2.91184352] rad

--- Estimation Errors ---
Freq Errors:  [-1.20742278  0.32840923 -0.82773776] Hz
Amp Errors:   [ 0.00832     0.00868313 -0.08303075]
Phase Errors: [ 0.37108816 -0.09285259 -6.02021763] rad


--- Running Root MUSIC ---

--- Estimation Results ---
Est Frequencies: [439.99865291 460.00630089 479.98387698] Hz
Est Amplitudes:  [1.41119875 1.40011987 1.44902883]
Est Phases:      [-3.04802373 -0.47552567  3.11305435] rad

--- Estimation Errors ---
Freq Errors:  [-0.00134709  0.00630089 -0.01612302] Hz
Amp Errors:   [ 0.00017094 -0.00040426 -0.0008162 ]
Phase Errors: [ 0.00051522 -0.00230938  0.00468023] rad


--- Running ESPRIT ---

--- Estimation Results ---
Est Frequencies: [440.15013961 459.96703565 480.02503654] Hz
Est Amplitudes:  [0.41823435 1.77925791 1.00002794]
Est Phases:      [-3.07635342 -0.69853686 -2.88152328] rad

--- Estimation Errors ---
Freq Errors:  [ 0.15013961 -0.03296435  0.02503654] Hz
Amp Errors:   [ 0.00284486  0.01359618 -0.0128905 ]
Phase Errors: [-0.02457284  0.00695845 -0.01390562] rad
```

(Note: The exact values for amplitudes, phases, and errors will vary due to their random generation.)

### Command-Line Options

You can customize the experiment via command-line arguments.

```bash
python main.py --freqs_true 440 445 450 --snr_db 25 --duration 0.8
```

To see all available options, run:

```bash
python main.py --help
```

| Argument| Description | Default |
| :-------- | :-------- | :-------- |
|`--fs`| Sampling frequency in Hz.| 44100.0 |
| `--duration` | Signal duration in seconds. | 0.1|
|`--snr_db` | Signal-to-Noise Ratio in dB. | 30.0|
| `--freqs_true`  | A list of true frequencies in Hz. | 440 460 480|
| `--amp_range` | The range for random amplitude generation. | 0.5 1.5|
| `--n_grids` | Number of grid points for Spectral MUSIC and Spectral Min-Norm. | 8192|
| `--sep_factor` | Separation factor for Root MUSIC, Root Min-Norm and ESPRIT. | 0.4|

### Using a Specific Analyzer in Your Own Code
The object-oriented design makes it easy to use any analyzer in your own projects. Here's how you might use the standard `SpectralMusicAnalyzer` and its enhanced `SpectralMusicAnalyzerFB` version:

```python
from analyzers.spectral import SpectralMusicAnalyzer, SpectralMusicAnalyzerFB
# ...

# Standard forward-only analyzer
spec_analyzer = SpectralMusicAnalyzer(fs=44100, n_sinusoids=3, n_grids=8192)
spec_analyzer.fit(my_signal)
estimated_freqs = spec_analyzer.frequencies

# Analyzer with Forward-Backward averaging for higher accuracy
spec_analyzer_fb = SpectralMusicAnalyzerFB(fs=44100, n_sinusoids=3, n_grids=8192)
spec_analyzer_fb.fit(my_signal)
accurate_freqs = spec_analyzer_fb.frequencies
```
In the same way, you can use the enhanced version (`...FB`) of `RootMusicAnalyzer`, `SpectralMinNormAnalyzer`, `RootMinNormAnalyzer`, `LSEspritAnalyzer`, and `TLSEspritAnalyzer`.

## Project Structure

This project is organized into a modular, object-oriented structure to promote clarity, reusability, and separation of concerns. The core logic is built upon a hierarchical class system.

-   `main.py`:
    The main entry point to run demonstrations. It orchestrates the setup, execution, and result presentation of the analysis.

-   `analyzers/`:
    A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
    -   `base.py`: Defines `AnalyzerBase`, the top-level abstract base class. It contains the common logic shared by *all* subspace-based methods, such as the `fit` method template, amplitude/phase estimation, and result properties.
    -   `music/`: A sub-package dedicated to the MUSIC algorithm and its variants.
        -   `base.py`: Defines `MusicAnalyzerBase`, an intermediate abstract class for all MUSIC variants. It inherits from `AnalyzerBase` and adds MUSIC-specific logic, like the estimation of the noise subspace.
        -   `spectral.py`: Implements `SpectralMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via spectral peak-picking.
        -   `root.py`: Implements `RootMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via polynomial rooting.
    -   `esprit/`: A sub-package dedicated to the ESPRIT algorithm and its variants.
        -   `base.py`:  Defines `EspritAnalyzerBase`, an intermediate abstract class for ESPRIT-based methods. It inherits from `AnalyzerBase`, and adds ESPRIT-specific logic, like the estimation of the signal subspace.
        -   `ls.py`: Implements `LSEspritAnalyzer` (inheriting from `EspritAnalyzerBase`), which uses the standard Least Squares approach to solve for the rotational operator.
        -   `tls.py`: Implements `TLSEspritAnalyzer` (inheriting from `EspritAnalyzerBase`), which uses the more robust Total Least Squares approach for higher accuracy in noisy conditions.
    -   `minnorm/`: A sub-package for Min-Norm algorithm variants.
        -   `base.py`: Defines `MinNormAnalyzerBase`, containing the core logic for computing the minimum norm vector.
        -   `spectral.py`: Implements `SpectralMinNormAnalyzer`, which estimates frequencies via spectral peak-picking.
        -   `root.py`: Implements `RootMinNormAnalyzer`, which estimates frequencies via polynomial rooting.
    -   `hoyw/`: A sub-package for the Higher-Order Yule-Walker (HOYW) method.
        -   `hoyw.py`: Implements `HOYWAnalyzer`, which directly inherits from `AnalyzerBase`. It estimates frequencies by solving the HOYW equations and subsequent finding the polynomial roots.
-   `mixins/`:
    A package for providing optional enhancements to the analyzer classes through multiple inheritance.
    -   `covariance.py`: Contains the `ForwardBackwardMixin` to add Forward-Backward averaging capability.

-   `utils/`:
    A package for reusable helper modules and data structures.
    -   `data_models.py`: Defines the `dataclass` structures (`SinusoidParameters`, `ExperimentConfig`).
    -   `signal_generator.py`: Provides functions for synthesizing test signals.

-   `cli.py`:
    A module dedicated to the Command-Line Interface. It handles argument parsing and the formatting of results for display.

This layered design allows for maximum code reuse and easy extension.

## Theoretical Background

The implemented methods are **model-based** high-resolution techniques that estimate sinusoidal parameters by fitting the observed signal to a predefined mathematical model. This approach allows for performance far exceeding that of traditional non-parametric methods like the FFT.

Two main families of models are explored in this project:

1.  **Subspace Models (MUSIC, ESPRIT, Min-Norm):** These methods model the signal's covariance matrix as having a low-rank signal component embedded in noise. They exploit the geometric properties of the signal and noise subspaces, which are obtained via eigenvalue decomposition.
2.  **Autoregressive (AR) Models (HOYW):** This approach models the signal as the output of a linear time-invariant system driven by white noise. Frequencies are estimated from the roots of the AR model's characteristic polynomial, whose coefficients are found from the signal's autocorrelation sequence.

For a deeper dive into the theory, please refer to the papers [1-3] for Spectral MUSIC, [4] for Root MUSIC, [5] for Min-Norm, [6] for ESPRIT, [7] for HOYW. 
The comprehensive textbook [8] provides detailed mathematical derivations and analyses of these methods and many other advanced signal processing techniques.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/tam17aki/music-esprit-python/blob/main/LICENSE) file for details.

## References
[1] Schmidt, R. O. (1979). “Multiple emitter location and signal parameter estimation,” in Proc. RADC, Spectral Estimation Workshop, Rome, NY, pp. 243–258.

[2] Bienvenu, G. (1979). “Influence of the spatial coherence of the background noise on high resolution passive methods,” in Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, Washington, DC, pp. 306–309.

[3] R.O. Schmidt, “Multiple emitter location and signal parameter estimation,” IEEE Trans. Antennas and Propagat., vol. AP-34, no. 3, pp. 276-280, 1986.

[4] A. Barabell, "Improving the resolution performance of eigenstructure-based direction-finding algorithms," ICASSP '83. IEEE International Conference on Acoustics, Speech, and Signal Processing, Boston, MA, USA, 1983, pp. 336-339, doi: 10.1109/ICASSP.1983.1172124.

[5]. R. Kumaresan and D. W. Tufts, "Estimating the Angles of Arrival of Multiple Plane Waves," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-19, no. 1, pp. 134-139, 1983.

[6] R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via rotational invariance techniques," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 7, pp. 984-995, 1989.

[7] P. Stoica, T. Soderstrom and F. Ti, "Asymptotic properties of the high-order Yule-Walker estimates of sinusoidal frequencies," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 11, pp. 1721-1734, 1989.

[8] P. Stoica and R. Moses, "Spectral Analysis of Signals," Pearson Prentice Hall, 2005.
