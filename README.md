# High-Resolution Parameter Estimation for Sinusoidal Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of high-resolution parameter estimation algorithms for sinusoidal signals including **MUSIC (Spectral/Root)** and **ESPRIT**. The project is structured with an object-oriented approach, emphasizing code clarity, reusability, and educational value.

This work is inspired by the foundational papers in subspace-based signal processing and aims to provide a practical and understandable guide to these powerful techniques.

## Features

- **High-Resolution Algorithms**: Implements spectral estimation techniques that surpass the resolution limits of the classical Fast Fourier Transform (FFT).
- **Multiple Methods Implemented**: A comprehensive suite of high-resolution algorithms is provided:
  - **MUSIC (Spectral & Root)**: A classic high-resolution method based on the orthogonality of signal and noise subspaces.
  - **Min-Norm (Spectral & Root)**: A variant of MUSIC that can reduce computational cost by using a single, optimized vector from the noise subspace.
  - **ESPRIT (Standard & Unitary)**: A computationally efficient method that estimates parameters directly without spectral search by exploiting rotational invariance.
    - The **Standard ESPRIT** (LS/TLS) provides a direct algebraic solution in the complex domain.
    - The **Unitary ESPRIT** (LS/TLS) variant transforms the problem into the real domain, significantly reducing computational complexity.
  - **HOYW**: A robust method based on the autocorrelation function and an AR model of the signal, enhanced with SVD-based rank truncation.
- **Full Parameter Estimation**: Not just frequencies, but also amplitudes and phases are estimated using a subsequent least-squares fit.
- **Object-Oriented Design**: Algorithms are encapsulated in clear, reusable classes with a consistent API, promoting clean code and extensibility.
- **Enhanced Accuracy Options**: Includes advanced techniques like **Forward-Backward Averaging** (via Mixins) and **Total Least Squares (TLS)** versions for most algorithms to improve performance in noisy conditions.
- **Clean and Type-Hinted Code**: Fully type-hinted with `mypy` validation. Code quality is enforced by a combination of `ruff` (for speed and formatting) and `pylint` (for in-depth analysis), ensuring high standards of maintainability.
- **Demonstration Script**: Includes a flexible command-line interface (`main.py`) to easily run experiments and compare the performance of different algorithms and their variants.

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
SNR:                30.0 dB
True Frequencies:   [440. 460. 480.] Hz
True Amplitudes:    [1.37608315 0.83629437 0.68580361]
True Phases:        [-2.57054399  3.06942632  1.2589388 ] rad

--- Running Spectral MUSIC ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
  N Grids: 16384

--- Estimation Results ---
Est Frequencies: [441.43066406 460.2722168  479.11376953] Hz
Est Amplitudes:  [1.31970983 0.91049428 0.63996752]
Est Phases:      [-3.02360917  3.01468944  1.60191168] rad

--- Estimation Errors ---
Freq Errors:  [ 1.43066406  0.2722168  -0.88623047] Hz
Amp Errors:   [-0.05637332  0.07419992 -0.04583609]
Phase Errors: [-0.45306517 -0.05473689  0.34297288] rad


--- Running Root MUSIC ---
Analyzer Parameters:
  Subspace Ratio: 0.3333

--- Estimation Results ---
Est Frequencies: [440.00253227 460.01181416 480.00302957] Hz
Est Amplitudes:  [1.3752451  0.83682708 0.68580222]
Est Phases:      [-2.57156131  3.06372891  1.25841805] rad

--- Estimation Errors ---
Freq Errors:  [0.00253227 0.01181416 0.00302957] Hz
Amp Errors:   [-8.38051531e-04  5.32715139e-04 -1.38207764e-06]
Phase Errors: [-0.00101732 -0.00569741 -0.00052075] rad


--- Running ESPRIT ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
  Solver: LSEspritSolver

--- Estimation Results ---
Est Frequencies: [440.00164814 460.00529982 480.00820167] Hz
Est Amplitudes:  [1.37552985 0.83680788 0.68588656]
Est Phases:      [-2.57136149  3.06593575  1.25639301] rad

--- Estimation Errors ---
Freq Errors:  [0.00164814 0.00529982 0.00820167] Hz
Amp Errors:   [-5.53302726e-04  5.13512704e-04  8.29517958e-05]
Phase Errors: [-0.0008175  -0.00349057 -0.0025458 ] rad
```

(Note: The exact values for amplitudes, phases, and errors will vary due to their random generation.)

### Command-Line Options

You can customize the experiment via command-line arguments.

```bash
python main.py --freqs_true 440 445 450 --snr_db 25 --duration 0.8 --complex
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
| `--subspace_ratio` | The ratio of the subspace dimension to the signal length.<br> Must be in (0, 0.5].| 1/3|
| `--complex` | If specified, generate a complex-valued signal instead of <br> a real-valued one.| False (Flag)|
| `--n_grids` | Number of grid points for Spectral MUSIC and Spectral Min-Norm. | 16384|


### Using a Specific Analyzer in Your Own Code
The object-oriented design makes it easy to use any analyzer in your own projects. Below are examples of how to use the different MUSIC and ESPRIT analyzers.

#### MUSIC Analyzers

```python
from analyzers.music.root import RootMusicAnalyzer, RootMusicAnalyzerFB
from analyzers.music.spectral import SpectralMusicAnalyzer, SpectralMusicAnalyzerFB
# ... assume 'my_signal' is a complex or float numpy array of your signal ...
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

- **`main.py`**: The main entry point to run demonstrations. It orchestrates the setup, execution, and result presentation of the analysis.
- **`analyzers/`**: A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
  - **`base.py`**: Defines `AnalyzerBase`, the top-level abstract base class for all parametric estimation methods. It contains the common logic for the analysis workflow, such as the `fit` method template, subsequent amplitude/phase estimation, and result properties.
  - **`models.py`**: Defines `TypedDict` models (e.g., `AnalyzerParameters`) used for structuring and type-hinting the dictionaries of hyperparameters passed to and returned by the analyzers.
  - **`music/`**: A sub-package dedicated to the MUSIC algorithm and its variants.
    - **`base.py`**: Defines `MusicAnalyzerBase`, an intermediate abstract class for all MUSIC variants. It inherits from `AnalyzerBase` and adds MUSIC-specific logic, like the estimation of the noise subspace.
    - **`spectral.py`**: Implements `SpectralMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via spectral peak-picking.
    - **`root.py`**: Implements `RootMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via polynomial rooting.
  - **`esprit/`**: A sub-package dedicated to the ESPRIT algorithm and its variants, including the computationally efficient Unitary ESPRIT.
    - **`base.py`**:  Defines `EspritAnalyzerBase`, an intermediate abstract class for ESPRIT-based methods. It inherits from `AnalyzerBase`, and adds ESPRIT-specific logic, like the estimation of the signal subspace.
    - **`standard.py`**: Implements `StandardEspritAnalyzer` for the classic, complex-valued ESPRIT algorithm.
    - **`unitary.py`**: Implements `UnitaryEspritAnalyzer`, which operates entirely on real-valued matrices for reduced computational load and improved accuracy.
    - **`solvers.py`**: Defines a set of solver classes that encapsulate the specific mathematical procedures for solving the ESPRIT core equations. This demonstrates the Strategy design pattern, allowing different numerical methods (LS, TLS, Unitary LS/TLS) to be flexibly injected into the analyzers.
  - **`minnorm/`**: A sub-package for Min-Norm algorithm variants.
    - **`base.py`**: Defines `MinNormAnalyzerBase`, containing the core logic for computing the minimum norm vector.
    - **`spectral.py`**: Implements `SpectralMinNormAnalyzer`, which estimates frequencies via spectral peak-picking.
    - **`root.py`**: Implements `RootMinNormAnalyzer`, which estimates frequencies via polynomial rooting.
  - **`hoyw/`**: A sub-package for the Higher-Order Yule-Walker (HOYW) method.
     - **`hoyw.py`**: Implements `HoywAnalyzer`, which directly inherits from `AnalyzerBase`. It estimates frequencies by solving the HOYW equations and subsequent finding the polynomial roots.
- **`mixins/`**: A package for providing optional enhancements to the analyzer classes through multiple inheritance.
  - **`covariance.py`**: Contains the `ForwardBackwardMixin` to add Forward-Backward averaging capability.
- **`utils/`**: A package for reusable helper modules and data structures that are decoupled from the specific analyzer implementations.
  - **`data_models.py`**: Defines the core `dataclass` models for the project.
    - `ExperimentConfig`: Encapsulates all parameters for a simulation run (e.g., SNR, duration), defining the "world" in which the signals exist.
    - `SinusoidParameters`: Represents the ground truth or estimated parameters of a signal, serving as the data "payload" that is generated and analyzed.
  - **`signal_generator.py`**: Provides functions for synthesizing test signals.
- **`cli.py`**: A module dedicated to the Command-Line Interface. It handles argument parsing and the formatting of results for display.

This layered design allows for maximum code reuse and easy extension.

## Architecture Overview

The project is built upon a flexible and extensible object-oriented architecture. The core of the library is a hierarchical system of analyzer classes, designed to maximize code reuse and clearly separate concerns.

The class diagram below illustrates the main **inheritance relationships** between the analyzer classes.

![Simple Class Diagram](https://github.com/tam17aki/music-esprit-python/blob/main/docs/images/simple_class_diagram.png)
*<div align="center">Fig. 1: Primary inheritance hierarchy of the analyzer classes</div>*

As shown, all analyzers inherit from a common `AnalyzerBase`, ensuring a consistent API. Specialized abstract classes like `MusicAnalyzerBase` and `EspritAnalyzerBase` group together logic common to each algorithm family.

### Key Design Patterns

Beyond this basic inheritance, the architecture leverages several key design patterns to add features and flexibility in a modular way.

-   **Strategy Pattern for Decoupling**: The ESPRIT algorithm's core numerical procedure is decoupled from the main analyzer class. The analyzers delegate this task to separate **`Solver`** objects (`LSEspritSolver`, `TLSUnitaryEspritSolver`, etc.), allowing different numerical solution strategies to be flexibly "plugged in."

-   **Mixin Classes for Feature Enhancement**: Optional features, such as Forward-Backward averaging, are added to concrete analyzers using **Mixin classes** (e.g., `ForwardBackwardMixin`). This allows for functionality to be added via composition, avoiding a rigid and deep inheritance tree.

-   **Structured Data Modeling**: The project heavily utilizes Python's typing features to create robust and self-documenting data structures.
    - **Analyzer's Public API (`get_params`)**: The `.get_params()` method returns a `TypedDict` model (`AnalyzerParameters`) instead of a plain dictionary. This provides a clear, type-safe structure for reporting the analyzers' hyperparameters.
    - **Analyzer's State (`est_params`)**: The estimation results are stored in an immutable `dataclass` object, `SinusoidParameters`. This encapsulates the output data (frequencies, amplitudes, phases) and ensures that the results, once computed, cannot be accidentally modified.

The complete architecture, including these mixin and composition relationships, is shown in the detailed class diagram below for those interested in the full implementation details.

![Complete Class Diagram](https://github.com/tam17aki/music-esprit-python/blob/main/docs/images/complete_class_diagram.png)
*<div align="center">Fig. 2: Detailed class diagram including Mixins, Solvers, and Data Models</div>* 

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
