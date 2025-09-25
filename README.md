# High-Resolution Parameter Estimation for Sinusoidal Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of high-resolution parameter estimation algorithms for sinusoidal signals including **MUSIC (Spectral/Root)** and **ESPRIT**. The project is structured with an object-oriented approach, emphasizing code clarity, reusability, and educational value.

This work is inspired by the foundational papers in subspace-based signal processing and aims to provide a practical and understandable guide to these powerful techniques.

## Features
- **Multiple Methods Implemented**:
  A comprehensive suite of advanced parameter estimation algorithms is provided, grouped by their core approach:
  - **High-Resolution Subspace and AR-Modeling Methods**: These techniques surpass the resolution limits of the classical FFT by exploiting the algebraic structure of the signal model.
    - **MUSIC (Spectral, Root, & FAST)**: A family of high-resolution methods based on the orthogonality of signal and noise subspaces.
       - The **Spectral** and **Root** MUSIC variants are classic implementations that offer true super-resolution capabilities.
       - The **FAST MUSIC** variant is a modern, computationally efficient implementation for (quasi-)periodic signals that replaces the expensive EVD with an FFT, prioritizing speed over ultimate resolution.
    - **Min-Norm (Spectral & Root)**: A variant of MUSIC that can reduce computational cost by using a single, optimized vector from the noise subspace.
    - **ESPRIT (Standard, Unitary, & FFT-based)**: A computationally efficient method that estimates parameters directly without spectral search by exploiting rotational invariance.
      - The **Standard** and **Unitary** variants provide high accuracy by computing the signal subspace via EVD/SVD.
      - The **FFT-ESPRIT** variant offers a significant speed-up ($`O(N\:\log\:N)`$) by approximating the signal subspace with an FFT-based kernel method, making it suitable for real-time applications.
    - **HOYW**: A robust method based on the autocorrelation function and an AR model of the signal, enhanced with SVD-based rank truncation.
  - **Fast Iterative Methods**:
    This approach prioritizes computational speed, making it ideal for applications where frequencies are well-separated.
    - **RELAX**: A greedy algorithm that estimates parameters sequentially. It iteratively finds the strongest signal component, subtracts it, and repeats the process on the residual signal, offering exceptional speed for well-separated sinusoids.
- **Full Parameter Estimation**: Not just frequencies, but also amplitudes and phases are estimated using a subsequent least-squares fit.
- **Object-Oriented Design**: Algorithms are encapsulated in clear, reusable classes with a consistent API, promoting clean code and extensibility.
- **Enhanced Accuracy Options**: Includes advanced techniques like **Forward-Backward Averaging** (via Mixins) and **Total Least Squares (TLS)** versions for most algorithms to improve performance in noisy conditions.
- **Clean and Type-Hinted Code**: Fully type-hinted with `mypy` validation. Code quality is enforced by a combination of `ruff` (for speed and formatting) and `pylint` (for in-depth analysis), ensuring high standards of maintainability.
- **Demonstration Script**: Includes a flexible command-line interface to easily run experiments and compare the performance of different algorithms and their variants.

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

     This is the recommended way to install the package for development or running examples. It installs the necessary dependencies and makes the package available throughout your environment.
    ```bash
    pip install -e .
    ```
    *(Note: This command reads the `pyproject.toml` file to understand how to install the package.)*

## Usage

After installation, you can run the provided example scripts to see the analyzers in action. These scripts are located in the `examples/` directory and serve as a starting point for your own experiments.

### Basic Example

To run a comparative analysis of MUSIC and ESPRIT with default parameters (a signal with three sinusoids at 440, 460, and 480 Hz):

```bash
python examples/run_comparison.py
```

#### Example Output
Running the command above will produce an output similar to this:

```
--- Experiment Setup ---
Sampling Frequency: 44100.0 Hz
Signal Duration:    100 ms
SNR:                30.0 dB
True Frequencies:   [440. 460. 480.] Hz
True Amplitudes:    [1.20169263 0.55724751 1.33230241]
True Phases:        [-1.04328679 -1.28130206  0.8902365 ] rad

--- Running Spectral MUSIC ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
  N Grids: 16384
Elapsed Time: 0.7471 seconds

--- Estimation Results ---
Est Frequencies: [440.29562612 460.26962506 479.18801542] Hz
Est Amplitudes:  [1.18226426 0.54646144 1.31235668]
Est Phases:      [-1.11323333 -1.2730893   1.13453565] rad

--- Estimation Errors ---
Freq Errors:  [ 0.29562612  0.26962506 -0.81198458] Hz
Amp Errors:   [-0.01942837 -0.01078606 -0.01994573]
Phase Errors: [-0.06994654  0.00821276  0.24429915] rad


--- Running Root MUSIC ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
Elapsed Time: 8.6092 seconds

--- Estimation Results ---
Est Frequencies: [440.01125456 460.01005004 480.00007988] Hz
Est Amplitudes:  [1.1997484  0.55909151 1.33171398]
Est Phases:      [-1.04718629 -1.28456043  0.89032596] rad

--- Estimation Errors ---
Freq Errors:  [1.12545575e-02 1.00500421e-02 7.98815982e-05] Hz
Amp Errors:   [-0.00194423  0.00184401 -0.00058843]
Phase Errors: [-3.89949770e-03 -3.25836839e-03  8.94516719e-05] rad


--- Running ESPRIT (LS) ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
  Solver: LSEspritSolver
Elapsed Time: 0.3819 seconds

--- Estimation Results ---
Est Frequencies: [440.00286966 460.00606773 480.00657837] Hz
Est Amplitudes:  [1.1999316  0.55884851 1.3318762 ]
Est Phases:      [-1.04475685 -1.28417663  0.88852574] rad

--- Estimation Errors ---
Freq Errors:  [0.00286966 0.00606773 0.00657837] Hz
Amp Errors:   [-0.00176103  0.00160101 -0.00042621]
Phase Errors: [-0.00147006 -0.00287457 -0.00171076] rad


--- Results Summary ---
Method         | Time (s) | Freq RMSE (Hz) | Amp RMSE | Phase RMSE (rad)
---------------|----------|----------------|----------|-----------------
Spectral MUSIC | 0.747131 | 0.522625       | 0.017240 | 0.146790        
Root MUSIC     | 8.609160 | 0.008712       | 0.001584 | 0.002934        
ESPRIT (LS)    | 0.381926 | 0.005426       | 0.001396 | 0.002110 
```

(Note: The exact values for amplitudes, phases, and errors will vary due to their random generation.)

### Running Focused Comparisons

For more specific comparisons, you can run other example scripts:

- `examples/compare_music_variants.py`:<br>
This script focuses exclusively on the MUSIC family, comparing the performance of `SpectralMusicAnalyzer`, `RootMusicAnalyzer`, and their Forward-Backward enhanced versions.

```bash
python examples/compare_music_variants.py
```

- `examples/compare_esprit_variants.py`:<br>
This script is dedicated to the ESPRIT family. It compares `StandardEspritAnalyzer` against the computationally efficient `UnitaryEspritAnalyzer`, with both LS and TLS solvers.

```bash
python examples/compare_esprit_variants.py
```

- `examples/compare_minnorm_variants.py`:<br>
This script is dedicated to the Min-Norm family, comparing the performance of `SpectralMinNormAnalyzer`, `RootMinNormAnalyzer`, and their Forward-Backward enhanced versions.

```bash
python examples/compare_minnorm_variants.py
```

You can customize the experiments by modifying these scripts or by using the command-line arguments they provide.

### Command-Line Options

You can customize the experiment via command-line arguments.

```bash
python examples/run_comparison.py --freqs_true 440 445 450 --snr_db 25 --duration 0.8 --complex
```

To see all available options, run:

```bash
python examples/run_comparison.py --help
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

## Project Structure

This project is organized into a modular, object-oriented structure to promote clarity, reusability, and separation of concerns. The core logic is built upon a hierarchical class system.

- **`analyzers/`**: A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
  - `base.py`: Defines `AnalyzerBase`, the top-level abstract base class for all parametric estimation methods. It contains the common logic for the analysis workflow, such as the `fit` method template, subsequent amplitude/phase estimation, and result properties.
  - `models.py`: Defines `TypedDict` models (e.g., `AnalyzerParameters`) used for structuring and type-hinting the dictionaries of hyperparameters passed to and returned by the analyzers.
  - **`music/`**: A sub-package dedicated to the MUSIC algorithm and its variants.
    - `base.py`: Defines `MusicAnalyzerBase`, an intermediate abstract class for all MUSIC variants. It inherits from `AnalyzerBase` and adds MUSIC-specific logic, like the estimation of the noise subspace.
    - `spectral.py`: Implements `SpectralMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via spectral peak-picking.
    - `root.py`: Implements `RootMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via polynomial rooting.
    - `fast.py`: Implements `FastMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies for periodic signals by analyzing the peaks of the ACF's power spectrum and computing a closed-form pseudospectrum.
  - **`esprit/`**: A sub-package dedicated to the ESPRIT algorithm and its variants, including the computationally efficient Unitary ESPRIT.
    - `base.py`:  Defines the abstract base classes for the ESPRIT family.
        -   `EspritAnalyzerBase`: The top-level base for all ESPRIT variants, providing common functionalities like frequency filtering.
        -   `EVDBasedEspritAnalyzer`: An intermediate base class for variants (like Standard and Unitary) that rely on computationally intensive Eigenvalue/Singular Value Decomposition (EVD/SVD) for subspace estimation.
    - `standard.py`: Implements `StandardEspritAnalyzer` (inheriting from `EVDBasedEspritAnalyzer`), the classic, complex-valued ESPRIT algorithm.
    - `unitary.py`: Implements `UnitaryEspritAnalyzer` (inheriting from `EVDBasedEspritAnalyzer`), which operates on real-valued matrices.
    - `fft.py`: Implements `FFTEspritAnalyzer`, a computationally efficient variant that approximates the signal subspace using an FFT-based kernel method instead of SVD/EVD.
    - `solvers.py`: Defines a set of solver classes that encapsulate the specific mathematical procedures for solving the ESPRIT core equations. This demonstrates the Strategy design pattern, allowing different numerical methods (LS, TLS, Unitary LS/TLS) to be flexibly injected into the analyzers.
  - **`minnorm/`**: A sub-package for Min-Norm algorithm variants.
    - `base.py`: Defines `MinNormAnalyzerBase`, containing the core logic for computing the minimum norm vector.
    - `spectral.py`: Implements `SpectralMinNormAnalyzer`, which estimates frequencies via spectral peak-picking.
    - `root.py`: Implements `RootMinNormAnalyzer`, which estimates frequencies via polynomial rooting.
  - **`hoyw/`**: A sub-package for the Higher-Order Yule-Walker (HOYW) method.
     - `hoyw.py`: Implements `HoywAnalyzer`, which directly inherits from `AnalyzerBase`. It estimates frequencies by solving the HOYW equations and subsequent finding the polynomial roots.
  - **`relax/`**: A sub-package for the RELAX algorithm.
     - `relax.py`: Implements `RelaxAnalyzer`, which sequentially estimates parameters using an iterative signal cancellation approach.
- **`mixins/`**: A package for providing optional enhancements to the analyzer classes through multiple inheritance.
  - `covariance.py`: Contains the `ForwardBackwardMixin` to add Forward-Backward averaging capability.
- **`utils/`**: A package for reusable helper modules and data structures that are decoupled from the specific analyzer implementations.
  - `data_models.py`: Defines the core `dataclass` models for the project.
    - `ExperimentConfig`: Encapsulates all parameters for a simulation run (e.g., SNR, duration), defining the "world" in which the signals exist.
    - `SinusoidParameters`: Represents the ground truth or estimated parameters of a signal, serving as the data "payload" that is generated and analyzed.
  - `signal_generator.py`: Provides functions for synthesizing test signals.
- **`cli.py`**: A module dedicated to the Command-Line Interface. It handles argument parsing and the formatting of results for display.
-  **`examples/`**: A directory containing example scripts that demonstrate how to use the library.
    - `run_comparison.py`: The main demonstration script that runs a comparative analysis of Spectral/Root MUSIC and ESPRIT algorithm. It can be used as a starting point for your own experiments.
    - `compare_music_variants.py`: The demonstration script that runs a comparative analysis of Spectral and Root MUSIC algorithm, including their Forward-Backward enhanced versions.
    - `compare_esprit_variants.py`: The demonstration script that runs a comparative analysis of Standard ESPRIT (LS/TLS) and  Unitary ESPRIT (LS/TLS) algorithm.
    - `compare_minnorm_variants.py`: The demonstration script that runs a comparative analysis of Spectral and Root Min-Norm algorithm, including their Forward-Backward enhanced versions.

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

1.  **Subspace Models (MUSIC, ESPRIT, Min-Norm):** These methods model the signal's covariance matrix as having a low-rank signal component embedded in noise. They exploit the geometric properties of the signal and noise subspaces, which are obtained via eigenvalue decomposition. For a detailed, step-by-step walkthrough of the MUSIC algorithm, please see the web version or download the PDF:
     - [Web Version (Markdown)](docs/theory/music_theory.md)
     - [Printable Version (PDF)](docs/theory/music_theory.pdf)
2.  **Autoregressive (AR) Models (HOYW):** This approach models the signal as the output of a linear time-invariant system driven by white noise. Frequencies are estimated from the roots of the AR model's characteristic polynomial, whose coefficients are found from the signal's autocorrelation sequence.

For a deeper dive into the theory behind each algorithm, please refer to the following key papers:

-   **MUSIC Variants**:
    -   Spectral MUSIC: [1-3]
    -   Root-MUSIC: [4]
    -   FAST MUSIC: [5]
-   **Min-Norm**: [6]
-   **ESPRIT Variants**:
    -   Standard ESPRIT: [7]
    -   Unitary ESPRIT: [8]
    -   FFT-ESPRIT: [9]
-   **HOYW**: [10]
-   **RELAX**: [11]

For a comprehensive overview and detailed mathematical derivations, the following textbook is highly recommended: [12].

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/tam17aki/music-esprit-python/blob/main/LICENSE) file for details.

## References
[1] Schmidt, R. O. (1979). “Multiple emitter location and signal parameter estimation,” in Proc. RADC, Spectral Estimation Workshop, Rome, NY, pp. 243–258.

[2] Bienvenu, G. (1979). “Influence of the spatial coherence of the background noise on high resolution passive methods,” in Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, Washington, DC, pp. 306–309.

[3] R.O. Schmidt, “Multiple emitter location and signal parameter estimation,” IEEE Trans. Antennas and Propagat., vol. AP-34, no. 3, pp. 276-280, 1986.

[4] A. Barabell, "Improving the resolution performance of eigenstructure-based direction-finding algorithms," in IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 8, pp. 336-339, 1983.

[5] O. Das, J. S. Abel, J. O. Smith III, "FAST MUSIC - An Efficient Implementation of The Music Algorithm For Frequency Estimation of Approximately Periodic Signals," International Conference on Digital Audio Effects (DAFx), vol.21, pp.342-348, 2018.

[6] R. Kumaresan and D. W. Tufts, "Estimating the Angles of Arrival of Multiple Plane Waves," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-19, no. 1, pp. 134-139, 1983.

[7] R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via rotational invariance techniques," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 7, pp. 984-995, 1989.

[8] M. Haardt and J. A. Nossek, "Unitary ESPRIT: how to obtain increased estimation accuracy with a reduced computational burden," in IEEE Transactions on Signal Processing, vol. 43, no. 5, pp. 1232-1242, 1995.

[9] S. L. Kiser, et al., "Fast Kernel-based Signal Subspace Estimates for Line Spectral Estimation," PREPRINT, 2023.

[10] P. Stoica, T. Soderstrom and F. Ti, "Asymptotic properties of the high-order Yule-Walker estimates of sinusoidal frequencies," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 11, pp. 1721-1734, 1989.

[11] J. Li and P. Stoica, "Efficient mixed-spectrum estimation with applications to target feature extraction," in IEEE Transactions on Signal Processing, vol. 44, no. 2, pp. 281-295, 1996.

[11] P. Stoica and R. Moses, "Spectral Analysis of Signals," Pearson Prentice Hall, 2005.
