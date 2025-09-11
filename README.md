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
True Amplitudes:    [1.22912604 0.94498725 1.05554323]
True Phases:        [-3.00626451 -1.41470502 -1.2889919 ] rad
SNR:                30.0 dB
Number of Grid Points:  16384

--- Running Spectral MUSIC ---

--- Estimation Results ---
Est Frequencies: [441.43066406 460.2722168  479.11376953] Hz
Est Amplitudes:  [1.19557359 0.99377398 1.05430486]
Est Phases:      [ 2.83499486 -1.58682732 -1.05501304] rad

--- Estimation Errors ---
Freq Errors:  [ 1.43066406  0.2722168  -0.88623047] Hz
Amp Errors:   [-0.03355245  0.04878673 -0.00123837]
Phase Errors: [ 5.84125937 -0.1721223   0.23397886] rad


--- Running ESPRIT ---

--- Estimation Results ---
Est Frequencies: [440.00119777 459.99668962 479.993247  ] Hz
Est Amplitudes:  [1.2304059  0.94458824 1.05422213]
Est Phases:      [-3.00528761 -1.41443859 -1.28755489] rad

--- Estimation Errors ---
Freq Errors:  [ 0.00119777 -0.00331038 -0.006753  ] Hz
Amp Errors:   [ 0.00127986 -0.00039902 -0.0013211 ]
Phase Errors: [0.0009769  0.00026643 0.00143701] rad
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
| `--n_grids` | Number of grid points for Spectral MUSIC and Spectral Min-Norm. | 16384|
| `--sep_factor` | Separation factor for Root MUSIC, Root Min-Norm and ESPRIT. | 0.4|

### Using a Specific Analyzer in Your Own Code
The object-oriented design makes it easy to use any analyzer in your own projects. Below are examples of how to use the different MUSIC and ESPRIT analyzers.

#### MUSIC Analyzers

```python
from analyzers.music.root import RootMusicAnalyzer, RootMusicAnalyzerFB
from analyzers.music.spectral import SpectralMusicAnalyzer, SpectralMusicAnalyzerFB
# ... assume 'my_signal' is a complex numpy array of your signal ...
# ... assume 'fs' and 'n_sinusoids' are defined ...

# Standard Spectral MUSIC analyzer
spec_analyzer = SpectralMusicAnalyzer(fs=fs, n_sinusoids=n_sinusoids)
spec_analyzer.fit(my_signal)
estimated_freqs = spec_analyzer.frequencies

# Standard Root MUSIC analyzer
root_analyzer = RootMusicAnalyzer(fs=fs, n_sinusoids=n_sinusoids)
root_analyzer.fit(my_signal)
estimated_freqs = root_analyzer.frequencies

# Spectral MUSIC with Forward-Backward averaging for higher accuracy
spec_analyzer_fb = SpectralMusicAnalyzerFB(fs=fs, n_sinusoids=n_sinusoids)
spec_analyzer_fb.fit(my_signal)
accurate_freqs = spec_analyzer_fb.frequencies

# Root MUSIC with Forward-Backward averaging for higher accuracy
root_analyzer_fb = RootMusicAnalyzerFB(fs=fs, n_sinusoids=n_sinusoids)
root_analyzer_fb.fit(my_signal)
accurate_freqs = root_analyzer_fb.frequencies
```

#### ESPRIT Analyzers

The ESPRIT analyzers utilize a Strategy design pattern, where the specific numerical solver (LS, TLS, etc.) is injected during initialization.

```python
from analyzers.esprit.solvers import (
    LSEspritSolver,
    LSUnitaryEspritSolver,
    TLSEspritSolver,
)
from analyzers.esprit.standard import StandardEspritAnalyzer
from analyzers.esprit.unitary import UnitaryEspritAnalyzer

# Standard ESPRIT with a Least Squares (LS) solver
ls_solver = LSEspritSolver()
ls_esprit_analyzer = StandardEspritAnalyzer(
    fs=fs, n_sinusoids=n_sinusoids, solver=ls_solver
)
ls_esprit_analyzer.fit(my_signal)
ls_freqs = ls_esprit_analyzer.frequencies

# Standard ESPRIT with a more robust Total Least Squares (TLS) solver
tls_solver = TLSEspritSolver()
tls_esprit_analyzer = StandardEspritAnalyzer(
    fs=fs, n_sinusoids=n_sinusoids, solver=tls_solver
)
tls_esprit_analyzer.fit(my_signal)
tls_freqs = tls_esprit_analyzer.frequencies

# Computationally efficient Unitary ESPRIT
unitary_solver = LSUnitaryEspritSolver()  # Or TLSUnitaryEspritSolver
unitary_esprit_analyzer = UnitaryEspritAnalyzer(
    fs=fs, n_sinusoids=n_sinusoids, solver=unitary_solver
)
unitary_esprit_analyzer.fit(my_signal)
unitary_freqs = unitary_esprit_analyzer.frequencies
```

(Note: The `...FB` classes for MUSIC and standard ESPRIT are created within their respective modules by inheriting from the `ForwardBackwardMixin`.)

## Project Structure

This project is organized into a modular, object-oriented structure to promote clarity, reusability, and separation of concerns. The core logic is built upon a hierarchical class system.

-   `main.py`:
    The main entry point to run demonstrations. It orchestrates the setup, execution, and result presentation of the analysis.

-   `analyzers/`:
    A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
    -   `base.py`: Defines `AnalyzerBase`, the top-level abstract base class for all parametric estimation methods. It contains the common logic for the analysis workflow, such as the `fit` method template, subsequent amplitude/phase estimation, and result properties.
    -   `music/`: A sub-package dedicated to the MUSIC algorithm and its variants.
        -   `base.py`: Defines `MusicAnalyzerBase`, an intermediate abstract class for all MUSIC variants. It inherits from `AnalyzerBase` and adds MUSIC-specific logic, like the estimation of the noise subspace.
        -   `spectral.py`: Implements `SpectralMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via spectral peak-picking.
        -   `root.py`: Implements `RootMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via polynomial rooting.
    -   `esprit/`: A sub-package dedicated to the ESPRIT algorithm and its variants, including the computationally efficient Unitary ESPRIT.
        -   `base.py`:  Defines `EspritAnalyzerBase`, an intermediate abstract class for ESPRIT-based methods. It inherits from `AnalyzerBase`, and adds ESPRIT-specific logic, like the estimation of the signal subspace.
        -   `standard.py`: Implements `StandardEspritAnalyzer` for the classic, complex-valued ESPRIT algorithm.
        -   `unitary.py`: Implements `UnitaryEspritAnalyzer`, which operates entirely on real-valued matrices for reduced computational load and improved accuracy.
        -   `solvers.py`: Defines a set of solver classes that encapsulate the specific mathematical procedures for solving the ESPRIT core equations. This demonstrates the Strategy design pattern, allowing different numerical methods (LS, TLS, Unitary LS/TLS) to be flexibly injected into the analyzers.
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

For a deeper dive into the theory, please refer to the papers [1-3] for Spectral MUSIC, [4] for Root MUSIC, [5] for Min-Norm, [6] for ESPRIT, [7] for Unitary ESPRIT, and [8] for HOYW. 
The comprehensive textbook [9] provides detailed mathematical derivations and analyses of these methods and many other advanced signal processing techniques.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/tam17aki/music-esprit-python/blob/main/LICENSE) file for details.

## References
[1] Schmidt, R. O. (1979). “Multiple emitter location and signal parameter estimation,” in Proc. RADC, Spectral Estimation Workshop, Rome, NY, pp. 243–258.

[2] Bienvenu, G. (1979). “Influence of the spatial coherence of the background noise on high resolution passive methods,” in Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, Washington, DC, pp. 306–309.

[3] R.O. Schmidt, “Multiple emitter location and signal parameter estimation,” IEEE Trans. Antennas and Propagat., vol. AP-34, no. 3, pp. 276-280, 1986.

[4] A. Barabell, "Improving the resolution performance of eigenstructure-based direction-finding algorithms," ICASSP '83. IEEE International Conference on Acoustics, Speech, and Signal Processing, Boston, MA, USA, 1983, pp. 336-339, doi: 10.1109/ICASSP.1983.1172124.

[5]. R. Kumaresan and D. W. Tufts, "Estimating the Angles of Arrival of Multiple Plane Waves," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-19, no. 1, pp. 134-139, 1983.

[6] R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via rotational invariance techniques," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 7, pp. 984-995, 1989.

[7] M. Haardt and J. A. Nossek, "Unitary ESPRIT: how to obtain increased estimation accuracy with a reduced computational burden," in IEEE Transactions on Signal Processing, vol. 43, no. 5, pp. 1232-1242, 1995.

[8] P. Stoica, T. Soderstrom and F. Ti, "Asymptotic properties of the high-order Yule-Walker estimates of sinusoidal frequencies," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 11, pp. 1721-1734, 1989.

[9] P. Stoica and R. Moses, "Spectral Analysis of Signals," Pearson Prentice Hall, 2005.
