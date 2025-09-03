# High-Resolution Parameter Estimation for Sinusoidal Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of high-resolution parameter estimation algorithms for sinusoidal signals including **MUSIC (Spectral/Root)**. The project is structured with an object-oriented approach, emphasizing code clarity, reusability, and educational value.

This work is inspired by the foundational papers in subspace-based signal processing and aims to provide a practical and understandable guide to these powerful techniques.

## Features

- **High-Resolution Algorithms**: Implements spectral estimation techniques that surpass the resolution limits of the classical Fast Fourier Transform (FFT).
- **Multiple Methods Implemented**:
  - **Spectral MUSIC**: Frequency estimation via spectral peak-picking.
  - **Root-MUSIC**: High-accuracy frequency estimation via polynomial rooting.
- **Full Parameter Estimation**: Not just frequencies, but also amplitudes and phases are estimated using a subsequent least-squares fit.
- **Enhanced Accuracy with Forward-Backward Averaging**: Improves estimation accuracy in low SNR or short data scenarios. This is implemented elegantly via a `ForwardBackwardMixin` class, showcasing a reusable and extensible design.
- **Demonstration Script**: Includes a command-line interface (`main.py`) to easily run experiments and compare the performance of different algorithms.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
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

The main entry point is `main.py`, which allows you to run a comparative analysis of the implemented algorithms.

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
| `--n_grids` | Number of grid points for Spectral MUSIC. | 8192|
| `--sep_factor` | Separation factor for Root-MUSIC. | 0.4|

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

In the same way, you can use the `RootMusicAnalyzer` and its enhanced `RootMusicAnalyzerFB`.

## Project Structure

This project is organized into a modular, object-oriented structure to promote clarity, reusability, and separation of concerns. The core logic is built upon a hierarchical class system.

-   **`main.py`**:
    The main entry point to run demonstrations. It orchestrates the setup, execution, and result presentation of the analysis.

-   **`analyzers/`**:
    A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
    -   **`base.py`**: Defines `AnalyzerBase`, the top-level abstract base class. It contains the common logic shared by *all* subspace-based methods, such as the `fit` method template, amplitude/phase estimation, and result properties.
    -   **`music/`**:
        -   **`base.py`**: Defines `MusicAnalyzerBase`, an intermediate abstract class for all MUSIC variants. It inherits from `AnalyzerBase` and adds MUSIC-specific logic, like the estimation of the noise subspace.
        -   **`spectral.py`**: Implements `SpectralMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via spectral peak-picking.
        -   **`root.py`**: Implements `RootMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via polynomial rooting.
  
-   **`mixins/`**:
    A package for providing optional enhancements to the analyzer classes through multiple inheritance.
    -   **`covariance.py`**: Contains the `ForwardBackwardMixin` to add Forward-Backward averaging capability.

-   **`utils/`**:
    A package for reusable helper modules and data structures.
    -   **`data_models.py`**: Defines the `dataclass` structures (`SinusoidParameters`, `ExperimentConfig`).
    -   **`signal_generator.py`**: Provides functions for synthesizing test signals.

-   **`cli.py`**:
    A module dedicated to the Command-Line Interface. It handles argument parsing and the formatting of results for display.

This layered design allows for maximum code reuse and easy extension. For instance, to add the ESPRIT algorithm, one would create an `EspritAnalyzerBase` inheriting from `AnalyzerBase`, and then implement concrete `EspritAnalyzer` classes, all while reusing the existing components.

## Theoretical Background

The implemented methods are based on the principle of subspace decomposition of a signal's covariance matrix. By performing an eigenvalue decomposition, the observation space can be separated into a **signal subspace** and an orthogonal **noise subspace**.
MUSIC leverages the orthogonality between the signal steering vectors and the noise subspace.

These techniques allow for the estimation of sinusoidal frequencies at a resolution far exceeding that of traditional methods like the FFT. For a deeper dive into the theory, please refer to the papers [1-3] for spectral-MUSIC, [4] for root-MUSIC.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References
[1] Schmidt, R. O. (1979). “Multiple emitter location and signal parameter estimation,” in Proc. RADC, Spectral Estimation Workshop, Rome, NY, pp. 243–258.

[2] Bienvenu, G. (1979). “Influence of the spatial coherence of the background noise on high resolution passive methods,” in Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, Washington, DC, pp. 306–309.

[3] R.O. Schmidt, “Multiple emitter location and signal parameter estimation,” IEEE Trans. Antennas and Propagat., vol. AP-34, no. 3, pp. 276-280, 1986.

[4] A. Barabell, "Improving the resolution performance of eigenstructure-based direction-finding algorithms," ICASSP '83. IEEE International Conference on Acoustics, Speech, and Signal Processing, Boston, MA, USA, 1983, pp. 336-339, doi: 10.1109/ICASSP.1983.1172124.
