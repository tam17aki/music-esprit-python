# High-Resolution Parameter Estimation for Sinusoidal Signals with MUSIC.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of high-resolution parameter estimation algorithms for sinusoidal signals with **MUSIC (Spectral/Root)**. The project is structured with an object-oriented approach, emphasizing code clarity, reusability, and educational value.

This work is inspired by the foundational papers in subspace-based signal processing and aims to provide a practical and understandable guide to these powerful techniques.

## Features

- **High-Resolution Algorithms**: Implements modern spectral estimation techniques that surpass the resolution limits of the classical Fast Fourier Transform (FFT).
- **Multiple Methods Implemented**:
  - **Spectral MUSIC**: Frequency estimation via spectral peak-picking.
  - **Root-MUSIC**: High-accuracy frequency estimation via polynomial rooting.
- **Full Parameter Estimation**: Not just frequencies, but also amplitudes and phases are estimated using a subsequent least-squares fit.
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


## Project Structure
The project is organized into modular components for clarity and reusability:

- `main.py`: The main script to run demonstrations.
- `analyzers/`: Contains the core algorithm implementations.
  - `base.py`: The abstract base class `MusicAnalyzerBase`.
  - `spectral.py`: The `SpectralMusicAnalyzer` class.
  - `root.py`: The `RootMusicAnalyzer` class.
- `utils/`: Contains helper modules.
  - `data_models.py`: Defines the dataclass structures for parameters and configuration.
  - `signal_generator.py`: Functions for synthesizing test signals.
- `cli.py`: Handles command-line argument parsing and result printing.

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
