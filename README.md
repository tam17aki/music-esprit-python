# High-Resolution Parameter Estimation for Sinusoidal Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of a comprehensive suite of modern, high-resolution parameter estimation algorithms for sinusoidal signals. It covers major algorithm families including **subspace-based methods (MUSIC, ESPRIT, Min-Norm, and their fast variants like FAST MUSIC and FFT-ESPRIT), AR-modeling (HOYW), and iterative greedy techniques (RELAX, CFH, NOMP).**

The project is architected with a clean, object-oriented design, emphasizing code clarity, reusability, and extensibility. It serves not only as a practical toolkit but also as an educational resource for understanding and comparing these powerful techniques. This work is inspired by the foundational papers in spectral estimation and aims to provide a robust and understandable guide.

## Features
- **Multiple Methods Implemented**:
  A comprehensive suite of advanced parameter estimation algorithms is provided, grouped by their core approach:
  - **High-Resolution Subspace and AR-Modeling Methods**: These techniques surpass the resolution limits of the classical FFT by exploiting the algebraic structure of the signal model.
    - **MUSIC (Spectral, Root, & FAST)**: A family of high-resolution methods based on the orthogonality of signal and noise subspaces.
       - The **Spectral** and **Root** **MUSIC** variants are classic implementations that offer true super-resolution capabilities.
       - The **FAST MUSIC** variant is a modern, computationally efficient implementation for (quasi-)periodic signals that replaces the expensive EVD with an FFT, prioritizing speed over ultimate resolution.
    - **Min-Norm (Spectral & Root)**: A variant of MUSIC that can reduce computational cost by using a single, optimized vector from the noise subspace.
    - **ESPRIT (Standard, Unitary, FFT-based, & Nyström-based)**: A computationally efficient method that estimates parameters directly without spectral search. It supports multiple numerical solvers configurable via a simple string argument (`solver="ls"` or `solver="tls"`).
      - The **Standard** and **Unitary** variants provide high accuracy by computing the signal subspace via EVD/SVD.
      - The **FFT-based** and **Nyström-based** variants offer significant speed-ups by approximating the signal subspace using different techniques (FFT kernels and matrix sampling, respectively).
    - **HOYW**: A robust method based on the autocorrelation function and an AR model of the signal, enhanced with SVD-based rank truncation.
  - **Fast Iterative Methods**:
    This approach prioritizes computational speed, making it ideal for applications where frequencies are well-separated.
    - **RELAX**: A classic greedy algorithm that identifies components by performing a high-density spectral search at each iteration. It uses a zero-padded FFT to achieve high accuracy, balancing speed and precision.
    - **CFH (Iterative DFT Interpolation)**: An extremely fast iterative method that replaces RELAX's dense search with a closed-form DFT interpolation. By using just three DFT samples, it offers one of the fastest estimation times and supports multiple interpolators (Candan, HAQSE) to trade speed for robustness.
      - **Multiple Interpolators**: Supports multiple 3-point interpolation strategies, allowing a trade-off between numerical robustness (**HAQSE/Serbes**) and computational simplicity (**Candan**).
    - **NOMP (Newtonized OMP)**: An advanced iterative method that incorporates a feedback mechanism. By cyclically re-refining all parameters, it can correct earlier estimates to achieve higher accuracy than forward-greedy methods.
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

### 1. Quick Start: The Main Comparison Demo

To get a quick overview of the main algorithm families, run the primary demonstration script:

```bash
python examples/run_comparison.py
```
This script compares a representative analyzer from each major family (MUSIC, ESPRIT, HOYW, etc.).

#### Example Output
Running the command above will produce an output similar to this:

```
--- Experiment Setup ---
Sampling Frequency: 44100.0 Hz
Signal Duration:    100 ms
SNR:                40.0 dB
True Frequencies:   [440. 445. 450.] Hz
True Amplitudes:    [1.05291351 1.23881033 1.18742655]
True Phases:        [-2.42839742 -2.94919381 -1.06740594] rad

...

--- Results Summary ---
Method          | Time (s) | Freq RMSE (Hz) | Amp RMSE | Phase RMSE (rad)
----------------|----------|----------------|----------|-----------------
Spectral MUSIC  | 0.615293 | 2.876581       | 0.523296 | 0.730882        
Root Min-Norm   | 1.065164 | 0.274875       | 0.084641 | 0.086743        
ESPRIT (LS)     | 0.264730 | 0.086608       | 0.054137 | 0.031641        
FFT-ESPRIT (LS) | 0.004962 | 0.135672       | 0.049331 | 0.032281        
HOYW            | 0.333640 | 98.257049      | 0.725682 | 0.880289        
RELAX           | 0.002581 | 6.894093       | 0.670390 | 1.000539   
```

*(Note: The exact values for amplitudes, phases, and errors will vary due to their random generation.)*

### 2. Focused Comparisons

While `run_comparison.py` provides a great overview of the main algorithm families, you may want to dive deeper into the specific trade-offs within each family. The following scripts are dedicated to these focused comparisons.

- `examples/compare_music_variants.py`:<br>This script focuses exclusively on the MUSIC family. It allows you to directly compare the performance and runtime of:
    -   Spectral MUSIC vs. Root-MUSIC vs. FAST MUSIC
    -   Standard vs. Forward-Backward enhanced versions (for Spectral/Root)
 ```bash
    python examples/compare_music_variants.py
 ```
- `examples/compare_standard_esprit.py`:<br>This script is dedicated to the high-accuracy variants of ESPRIT family, comparing the trade-offs between:
    -   Standard ESPRIT vs. Unitary ESPRIT vs. Forward-Backward enhanced versions (for Standard)
    -   Least Squares (`solver="ls"`) vs. Total Least Squares (`solver="tls"`) solvers
 ```bash
    python examples/compare_standard_esprit.py
 ```
- `examples/compare_fast_esprit.py`:<br>This script focuses on the computationally efficient, approximation-based variants of the ESPRIT family. It allows you to compare the trade-offs between:
    -   Subspace Approximation Methods:
        -   Nyström-based ESPRIT vs. FFT-based ESPRIT
    -   Numerical Solvers:
        -   Least Squares (`solver="ls"`) vs. Total Least Squares (`solver="tls"`) solvers for both methods.
        -   For FFT-ESPRIT, an even faster Woodbury-based LS (`solver="woodbury"`) is also available for comparison.
    ```bash
    python examples/compare_fast_esprit.py
    ```
- `examples/compare_minnorm_variants.py`:<br>This script explores the Min-Norm family, comparing:
    -   Spectral Min-Norm vs. Root Min-Norm
    -   Standard vs. Forward-Backward enhanced versions
 ```bash
    python examples/compare_minnorm_variants.py
 ```
- `examples/compare_iterative_methods.py`:<br>This script is dedicated to the fast iterative methods, allowing for a direct comparison of:
    - RELAX (using a dense zero-padded FFT search)
    - CFH with the HAQSE/Serbes interpolator
    - CFH with the Candan interpolator
    - NOMP (Newtonized Orthogonal Matching Pursuit)
```bash
   python examples/compare_iterative_methods.py
```

These scripts are the best place to understand the subtle but important differences between the various implementations provided in this library.

### 3. Comprehensive Benchmark: The All-in-One Script

For an exhaustive comparison of **all** implemented analyzers and their variants (including LS/TLS solvers and Forward-Backward versions), run the comprehensive benchmark script:

```bash
   python examples/all_in_one_demo.py
```

*(Note: This script may take longer to run as it executes every available combination.)*

### 4. Command-Line Options

All of the above experiments can be easily customized using a shared set of command-line arguments. This allows you to test the algorithms' performance with different signals and noise conditions.

For example, to test with closely spaced frequencies at a lower SNR:

```bash
python examples/run_comparison.py --freqs_true 440 445 450 --snr_db 20
```

To see all available options, run:

```bash
python examples/run_comparison.py --help
```

| Argument| Description | Default |
| :-------- | :-------- | :-------- |
|`--fs`| Sampling frequency in Hz.| `44100.0` |
| `--duration` | Signal duration in seconds. | `0.1`|
|`--snr_db` | Signal-to-Noise Ratio in dB. | `30.0`|
| `--freqs_true`  | List of true frequencies in Hz (space separated). | `440.0 460.0 480.0`|
| `--amp_range` | Range for random generation of sinusoidal amplitudes. | `0.5 1.5`|
| `--subspace_ratio` | Ratio of the subspace dimension to the signal length.<br>Must be in (0, 0.5].| `1/3`|
| `--complex` | If specified, generate a complex-valued signal instead <br>of a real-valued one.| `False` (Flag)|
| `--n_grids` | Number of grid points for Spectral MUSIC and Spectral<br>Min-Norm method. | `16384`|
| `--min_freq_period`| Minimum frequency for periodicity search for FAST <br>MUSIC method. | `20.0`|
| `--ar_order` | Order of the AutoRegressive (AR) model for HOYW<br>method. | `512`|
| `--rank_factor` | Factor to determine the number of rows to sample for<br>Nyström-based ESPRIT method. | `10`|
| `--n_fft_iip`	| FFT length for iterative methods (RELAX, FFT-ESPRIT).<br>If not specified, defaults to the signal length.| `None`|
| `--cfh_interpolator` | Interpolator method for the CFH analyzer. Can be<br>`candan` or `haqse`. | `haqse` |
| `--n_newton_steps` | Number of Newton refinement steps for NOMP. | `1` |
| `--n_cyclic_rounds`| Number of cyclic refinement rounds for NOMP. | `1` |
| `--nomp_conv_thresh` | Convergence threshold for NOMP's cyclic refinement.<br>Set to 0 to disable and run for fixed rounds. | `1e-6` |

## Project Structure

This project is organized into a modular, object-oriented structure to promote clarity, reusability, and separation of concerns. The core logic is built upon a hierarchical class system.

- **`analyzers/`**: A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
  - `base.py`: Defines `AnalyzerBase`, the top-level abstract base class for all parametric estimation methods. It contains the common logic for the analysis workflow, such as the `fit` method template, subsequent amplitude/phase estimation, and result properties.
  - `models.py`: Defines `TypedDict` models (e.g., `AnalyzerParameters`) used for structuring and type-hinting the dictionaries of hyperparameters passed to and returned by the analyzers.
  - `factory.py`: Provides a set of factory functions (e.g., `get_music_analyzers`) that encapsulate the logic for instantiating the various analyzer classes. This simplifies the setup process in the example scripts.**
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
    - `nystrom.py`: Implements `NystromEspritAnalyzer`, a fast variant that approximates the signal subspace using the Nyström method, reducing the complexity of the EVD step.
    - `fft.py`: Implements `FFTEspritAnalyzer`, a computationally efficient variant that approximates the signal subspace using an FFT-based kernel method instead of SVD/EVD.
    - `solvers.py`: Defines a set of solver functions that encapsulate the specific mathematical procedures for solving the ESPRIT core equations. This demonstrates a Pythonic implementation of the Strategy design pattern using functions as first-class objects. To ensure type safety, this module also defines `Callable` type aliases for the different solver function signatures and `TypedDict` models to structure the collections of available solvers.
  - **`minnorm/`**: A sub-package for Min-Norm algorithm variants.
    - `base.py`: Defines `MinNormAnalyzerBase`, containing the core logic for computing the minimum norm vector.
    - `spectral.py`: Implements `SpectralMinNormAnalyzer`, which estimates frequencies via spectral peak-picking.
    - `root.py`: Implements `RootMinNormAnalyzer`, which estimates frequencies via polynomial rooting.
  - **`hoyw/`**: A sub-package for the Higher-Order Yule-Walker (HOYW) method.
     - `hoyw.py`: Implements `HoywAnalyzer`, which directly inherits from `AnalyzerBase`. It estimates frequencies by solving the HOYW equations and subsequent finding the polynomial roots.
  - **`iterative/`**: A sub-package for fast iterative greedy methods.
    - `base.py`: Defines `IterativeAnalyzerBase`, an intermediate abstract class that provides the common signal cancellation framework used by RELAX and CFH.
    - `relax.py`: Implements `RelaxAnalyzer` (inheriting from `IterativeAnalyzerBase`), which uses a high-density zero-padded FFT search to find signal components.
    - `cfh.py`: Implements `CfhAnalyzer` (inheriting from `IterativeAnalyzerBase`), which uses a fast, closed-form DFT interpolation method (Candan or HAQSE) to find components.
    - `nomp.py`: Implements `NompAnalyzer`, a more advanced iterative method featuring Newton-based refinements and a cyclic feedback loop.
- **`mixins/`**: A package for providing optional enhancements to the analyzer classes through multiple inheritance.
  - `covariance.py`: Contains the `ForwardBackwardMixin` to add Forward-Backward averaging capability.
- **`utils/`**: A package for reusable helper modules and data structures that are decoupled from the specific analyzer implementations.
  - `data_models.py`: Defines the core `dataclass` models for the project.
    - `ExperimentConfig`: Encapsulates all parameters for a simulation run (e.g., SNR, duration), defining the "world" in which the signals exist.
    - `AlgorithmConfig`: Encapsulates hyperparameters for the analyzer algorithms (e.g., subspace ratio, number of grids), defining the configuration of the "tools".
    - `SinusoidParameters`: Represents the ground truth or estimated parameters of a signal, serving as the data "payload" that is generated and analyzed.
  - `signal_generator.py`: Provides functions for synthesizing test signals.
- **`cli.py`**: A module dedicated to the Command-Line Interface. It handles argument parsing and the formatting of results for display.
-  **`examples/`**: A directory containing example scripts that demonstrate how to use the library.
    - `run_comparison.py`: The main demonstration script that runs a comparative analysis of all major algorithm families.
    - `compare_music_variants.py`: The demonstration script that runs a comparative analysis of Spectral MUSIC, Root MUSIC and FAST MUSIC algorithm, including their Forward-Backward enhanced versions (for Spectral/Root).
    - `compare_standard_esprit.py`: The demonstration script that runs a comparative analysis of Standard ESPRIT (LS/TLS) and Unitary ESPRIT (LS/TLS) algorithm.
    - `compare_fast_esprit.py`: The demonstration script that runs a comparative analysis of Nyström-based ESPRIT (LS/TLS) and FFT-ESPRIT (LS/TLS) algorithm.
    - `compare_minnorm_variants.py`: The demonstration script that runs a comparative analysis of Spectral and Root Min-Norm algorithm, including their Forward-Backward enhanced versions.
    - `compare_iterative_methods.py`: The demonstration script that runs a focused comparison of the fast iterative methods: RELAX vs. the different variants of CFH (Candan, HAQSE).
    - `all_in_one_demo.py`: A comprehensive benchmark script that runs an exhaustive comparison of **all** implemented analyzers and their variants. Ideal for thorough performance evaluation and regression testing.

This layered design allows for maximum code reuse and easy extension.

## Architecture Overview

The project is built upon a flexible and extensible object-oriented architecture. The core of the library is a hierarchical system of analyzer classes, designed to maximize code reuse and clearly separate concerns.

The class diagram below illustrates the main **inheritance relationships** between the analyzer classes.

![Simple Class Diagram](https://github.com/tam17aki/music-esprit-python/blob/main/docs/images/simple_class_diagram.png)
*<div align="center">Fig. 1: Primary inheritance hierarchy of the analyzer classes</div>*

As shown, all analyzers inherit from a common `AnalyzerBase`, ensuring a consistent API. Specialized abstract classes like `MusicAnalyzerBase`, `EspritAnalyzerBase`, and `IterativeAnalyzerBase` group together logic common to each algorithm family. For instance, `IterativeAnalyzerBase` encapsulates the sequential signal cancellation workflow, allowing subclasses like `RelaxAnalyzer` and `CfhAnalyzer` to focus solely on their unique strategy for finding the next strongest signal component.

### Key Design Patterns

Beyond this basic inheritance, the architecture leverages several key design patterns to add features and flexibility in a modular way.

-   **Strategy Pattern with First-Class Functions**: The core numerical procedure of the ESPRIT algorithm is decoupled into a set of separate **solver functions** (`solve_esprit_ls`, `solve_unitary_esprit_tls`, etc.). The main analyzer classes (`StandardEspritAnalyzer`, etc.) act as the "Context" that holds a reference to one of these functions. Based on a simple string argument provided at initialization, the analyzer selects the appropriate solver function to use, demonstrating a Pythonic implementation of the Strategy pattern that leverages functions as first-class objects. This allows different numerical strategies to be flexibly chosen without exposing implementation details to the user.

-   **Mixin Classes for Feature Enhancement**: Optional features, such as Forward-Backward averaging, are added to concrete analyzers using **Mixin classes** (e.g., `ForwardBackwardMixin`). This allows for functionality to be added via composition, avoiding a rigid and deep inheritance tree.

-   **Structured Data Modeling**: The project heavily utilizes Python's typing features to create robust and self-documenting data structures.
    - **Analyzer's Public API (`get_params`)**: The `.get_params()` method returns a `TypedDict` model (`AnalyzerParameters`) instead of a plain dictionary. This provides a clear, type-safe structure for reporting the analyzers' hyperparameters.
    - **Analyzer's State (`est_params`)**: The estimation results are stored in an immutable `dataclass` object, `SinusoidParameters`. This encapsulates the output data (frequencies, amplitudes, phases) and ensures that the results, once computed, cannot be accidentally modified.

The complete architecture, including these mixin and composition relationships, is shown in the detailed class diagram below for those interested in the full implementation details.

![Complete Class Diagram](https://github.com/tam17aki/music-esprit-python/blob/main/docs/images/complete_class_diagram.png)
*<div align="center">Fig. 2: Detailed class diagram including Mixins, Solvers, and Data Models</div>* 

## Theoretical Background

The implemented methods are **model-based** high-resolution techniques that estimate sinusoidal parameters by fitting the observed signal to a predefined mathematical model. This approach allows for performance far exceeding that of traditional non-parametric methods like the FFT.

Three main families of models are explored in this project:

1.  **Subspace Models (MUSIC, ESPRIT, Min-Norm):** These methods model the signal's covariance matrix as having a low-rank signal component embedded in noise. The key insight is that this matrix can be decomposed via eigenvalue decomposition into two orthogonal subspaces: a *signal subspace* spanned by the eigenvectors corresponding to the largest eigenvalues, and a *noise subspace* spanned by the remaining eigenvectors. The different algorithms in this family exploit the geometric properties of these subspaces in unique ways.
    *   The **MUSIC** and **Min-Norm** algorithms are based on the **orthogonality principle**: the steering vectors of the true sinusoidal components are orthogonal to the noise subspace. They estimate frequencies by performing a spectral search for vectors that satisfy this orthogonality condition, resulting in a pseudospectrum with sharp peaks at the signal frequencies.
    *   The **ESPRIT** algorithm is based on the **rotational invariance principle**. It exploits a special shift-invariant structure within the signal subspace itself. By solving a small generalized eigenvalue problem, it can directly calculate the frequencies without performing an expensive spectral search, making it computationally efficient.
    *   For a detailed, step-by-step walkthrough of the MUSIC algorithm, please see the web version or download the PDF:
        *   [Web Version (Markdown)](docs/theory/music_theory.md)
        *   [Printable Version (PDF)](docs/theory/music_theory.pdf)

2.  **Autoregressive (AR) Models (HOYW):** This approach models the signal as the output of a linear time-invariant system driven by white noise. The AR model coefficients are determined from the signal's autocorrelation sequence using the Higher-Order Yule-Walker (HOYW) equations. Frequencies are then estimated from the roots of the AR model's characteristic polynomial that lie on or near the unit circle.

3.  **Iterative Greedy Methods (RELAX, CFH, NOMP):** This approach estimates parameters sequentially, one component at a time. It "greedily" finds the strongest sinusoidal component in the signal, subtracts it to form a residual signal, and then repeats the process on the residual. This method can be exceptionally fast for well-separated sinusoids.
    *   The **RELAX** and **CFH** algorithms are forward-greedy methods, where estimates from previous iterations are fixed. They differ in their strategy for finding the next component (dense zero-padded FFT search vs. DFT interpolation). In this library, they are implemented as subclasses of a common iterative framework.
    *   The **NOMP** algorithm builds upon this greedy framework by incorporating a **feedback mechanism**. Through cyclic Newton-based refinements and least-squares updates, it re-evaluates all prior estimates after each new component is detected, allowing it to correct for inter-component interference and achieve higher accuracy.

For a deeper dive into the theory behind each algorithm, please refer to the following key papers:

-   **MUSIC Variants**:
    -   Spectral MUSIC: [1-3]
    -   Root-MUSIC: [4]
    -   FAST MUSIC: [5]
-   **Min-Norm**: [6]
-   **ESPRIT Variants**:
    -   Standard ESPRIT: [7]
    -   Unitary ESPRIT: [8]
    -   Nyström-based ESPRIT: [9]
    -   FFT-ESPRIT: [10]
-   **HOYW**: [11]
-   **RELAX / CFH / NOMP**:
    -   RELAX: [12]
    -   DFT Interpolation (Candan): [13]
    -   DFT Interpolation (Serbes/HAQSE): [14]
    -   NOMP: [15]

For a comprehensive overview and detailed mathematical derivations, the following textbook is highly recommended: [16].

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

[9] C. Qian, L. Huang, and H.C. So, "Computationally efficient ESPRIT algorithm for direction-of-arrival estimation based on Nyström method," Signal Processing, vol. 94, pp. 74-80, 2014.

[10] S. L. Kiser, et al., "Fast Kernel-based Signal Subspace Estimates for Line Spectral Estimation," PREPRINT, 2023.

[11] P. Stoica, T. Soderstrom and F. Ti, "Asymptotic properties of the high-order Yule-Walker estimates of sinusoidal frequencies," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 11, pp. 1721-1734, 1989.

[12] J. Li and P. Stoica, "Efficient mixed-spectrum estimation with applications to target feature extraction," in IEEE Transactions on Signal Processing, vol. 44, no. 2, pp. 281-295, 1996.

[13] C. Candan, "A method for fine resolution frequency estimation from three DFT samples," IEEE Signal Processing Letters, vol. 18, no. 6, pp. 351-354, 2011.

[14] A. Serbes, "Fast and efficient sinusoidal frequency estimation by using the DFT coefficients," IEEE Transactions on Communications, vol. 67, no. 3, pp. 2333-2342, 2019.

[15] B. Mamandipoor, D. Ramasamy and U. Madhow, "Newtonized Orthogonal Matching Pursuit: Frequency Estimation Over the Continuum," in IEEE Transactions on Signal Processing, vol. 64, no. 19, pp. 5066-5081, 2016.

[16] P. Stoica and R. Moses, "Spectral Analysis of Signals," Pearson Prentice Hall, 2005.
