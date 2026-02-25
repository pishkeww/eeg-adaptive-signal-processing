"""
EEG Artifact Removal using LMS Adaptive Filtering
--------------------------------------------------
Uses a reference EOG channel to cancel ocular artifacts
from EEG recordings via sample-by-sample LMS adaptation.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample


data_path = sample.data_path()
raw = mne.io.read_raw_fif(
    data_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=True, verbose=False
)

# Keep only EEG and EOG channels
raw.pick_types(eeg=True, eog=True)
raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)

sfreq = raw.info['sfreq']
data, times = raw.get_data(return_times=True)

# Identify one EEG and one EOG channel
eog_idx = mne.pick_types(raw.info, eog=True)[0]
eeg_idx = mne.pick_types(raw.info, eeg=True)[0]

eog_signal = data[eog_idx]   # reference (artifact source)
eeg_signal = data[eeg_idx]   # primary (contaminated signal)

# Normalise for numerical stability
eog_signal = eog_signal / np.max(np.abs(eog_signal))
eeg_signal = eeg_signal / np.max(np.abs(eeg_signal))
# LMS
def lms_filter(primary, reference, mu=0.01, filter_order=32):
    """
    LMS adaptive filter.
    primary   : contaminated EEG channel (desired signal + artifact)
    reference : EOG channel (artifact reference)
    mu        : step size (learning rate)
    filter_order : number of filter taps
    Returns cleaned signal and weight convergence history.
    """
    n_samples = len(primary)
    weights = np.zeros(filter_order)
    output = np.zeros(n_samples)
    error  = np.zeros(n_samples)
    weight_history = np.zeros((n_samples, filter_order))

    for n in range(filter_order, n_samples):
        # Reference signal buffer (most recent samples)
        x = reference[n - filter_order:n][::-1]

        # Filter output: estimated artifact
        y = np.dot(weights, x)

        # Error: cleaned EEG signal
        e = primary[n] - y

        # LMS weight update
        weights += 2 * mu * e * x

        output[n] = y
        error[n]  = e
        weight_history[n] = weights

    return error, output, weight_history


cleaned, artifact_estimate, w_history = lms_filter(
    eeg_signal, eog_signal, mu=0.005, filter_order=32
)

# ── 3. Quantify artifact suppression ─────────────────────────────────────────
# Compare RMS of signal before and after in a window after filter settles
settle = 500  # skip first samples while filter converges
rms_before = np.sqrt(np.mean(eeg_signal[settle:] ** 2))
rms_after  = np.sqrt(np.mean(cleaned[settle:] ** 2))
suppression_db = 20 * np.log10(rms_before / rms_after)
print(f"RMS before : {rms_before:.4f}")
print(f"RMS after  : {rms_after:.4f}")
print(f"Suppression: {suppression_db:.2f} dB")


fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

t = times[:5000]

# Raw EEG
axes[0].plot(t, eeg_signal[:5000], color='steelblue', linewidth=0.8)
axes[0].set_title('Original EEG (contaminated)')
axes[0].set_ylabel('Amplitude (norm.)')

# EOG reference
axes[1].plot(t, eog_signal[:5000], color='tomato', linewidth=0.8)
axes[1].set_title('EOG Reference (artifact source)')
axes[1].set_ylabel('Amplitude (norm.)')

# Cleaned EEG
axes[2].plot(t, cleaned[:5000], color='seagreen', linewidth=0.8)
axes[2].set_title('Cleaned EEG (after LMS artifact removal)')
axes[2].set_ylabel('Amplitude (norm.)')
axes[2].set_xlabel('Time (s)')

# Weight convergence
axes[3].plot(np.sqrt(np.sum(w_history**2, axis=1))[:5000],
             color='darkorchid', linewidth=0.8)
axes[3].set_title('Filter Weight Norm (convergence)')
axes[3].set_ylabel('||w||')
axes[3].set_xlabel('Sample index')

plt.tight_layout()
plt.savefig('eeg_lms_artifact_removal.png', dpi=150)
plt.show()
print("Plot saved.")