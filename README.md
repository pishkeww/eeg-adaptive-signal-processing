# EEG Adaptive Signal Processing: VEP Characterization Toolkit

This repository contains a framework for the denoising, decomposition, and feature extraction of **Visual Evoked Potential (VEP)** signals. It addresses the inherent challenges of EEG data, specifically the low signal-to-noise ratio (SNR) and non-stationary nature of neural responses.

The project implements advanced mode decomposition and entropy-based measures to identify neurophysiological biomarkers more effectively than traditional Fourier-based analysis.

---

## 🛠 Core Methodology

The processing pipeline is built on three pillars of advanced signal analysis:

* **Signal Decomposition (`decomposition.py`)**: Implements techniques such as **Empirical Mode Decomposition (EMD)** or **Variational Mode Decomposition (VMD)** to break down complex EEG signals into their constituent Intrinsic Mode Functions (IMFs).
* **Adaptive Filtering (`adaptive_filter.py`)**: Utilizes adaptive algorithms to suppress ocular artifacts and power-line interference without distorting the underlying VEP components.
* **Complexity Analysis (`entropy_metrics.py`)**: Employs **Multiscale Fluctuation-Based Dispersion Entropy** to quantify the irregularity and dynamical complexity of the decomposed signals.



---

## Key Research Findings

* **Beyond FFT**: Demonstrated that mode decomposition captures transient neural events that are often "smeared" in traditional Power Spectral Density (PSD) analysis.
* **Sensitivity**: The integration of dispersion entropy provided a higher sensitivity in distinguishing between baseline and stimulus-evoked states compared to standard variance-based metrics.
* **Clinical Potential**: The framework is designed to aid in the automated diagnosis of neurological irregularities by providing objective, numerical biomarkers.

---

## Technical Stack

* **Language**: Python
* **Signal Processing**: NumPy, SciPy, PyEMD (or equivalent decomposition libraries).
* **Mathematics**: Information Theory (Entropy), Stochastic Calculus, and Time-Frequency Analysis.

---
*Note: This repository is part of a research project on advancing the characterization of neurological responses using nonlinear dynamics.*
