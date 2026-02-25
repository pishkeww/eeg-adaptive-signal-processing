"""
Cognitive State Kalman Filter from EEG (EEGBCI Dataset)
---------------------------------------------------------
Estimates latent motor imagery state from EEG using mu/beta
band suppression over motor cortex channels (C3, Cz, C4).

Dataset: EEGBCI (built into MNE, downloads automatically)
- Rest condition  → baseline (label 0)
- Motor imagery   → active state (label 1)

Neurophysiology:
    Mu band (8-12 Hz) and Beta band (13-30 Hz) suppress
    during motor imagery over motor cortex — this is the
    correct feature for this dataset.

State-space model:
    x_{t+1} = A * x_t + w_t      w ~ N(0, Q)
    y_t     = C * x_t + v_t      v ~ N(0, R)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import accuracy_score, roc_auc_score
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


subject   = 3
runs_rest = [1]   # eyes open rest
runs_task = [3]   # motor imagery left/right hand

print("Loading EEGBCI data...")
raw_rest = concatenate_raws([
    read_raw_edf(f, preload=True, verbose=False)
    for f in eegbci.load_data(subject, runs_rest)
])
raw_task = concatenate_raws([
    read_raw_edf(f, preload=True, verbose=False)
    for f in eegbci.load_data(subject, runs_task)
])

for raw in [raw_rest, raw_task]:
    eegbci.standardize(raw)
    raw.pick_types(eeg=True)
    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()

sfreq = raw_rest.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")

# Select Motor Cortex Channels 
# Motor imagery signal strongest at C3, Cz, C4
motor_channels = ['C3', 'Cz', 'C4']

def get_motor_data(raw):
    available = [ch for ch in motor_channels if ch in raw.ch_names]
    print(f"Using motor cortex channels: {available}")
    picks = mne.pick_channels(raw.ch_names, include=available)
    return raw.get_data(picks=picks), available

data_rest, ch_used = get_motor_data(raw_rest)
data_task, _       = get_motor_data(raw_task)

#  Feature Extraction: Mu/Beta Suppression 
def band_power(signal, sfreq, band, nperseg=128):
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx])

def extract_motor_feature(eeg_window, sfreq):
    """
    Motor imagery suppresses mu (8-12 Hz) and beta (13-30 Hz).
    Feature = negative mean power in these bands over motor channels.
    Higher feature value = more suppression = more motor imagery.
    """
    mus, betas = [], []
    for ch in range(eeg_window.shape[0]):
        mus.append(band_power(eeg_window[ch], sfreq, [8, 12]))
        betas.append(band_power(eeg_window[ch], sfreq, [13, 30]))
    # Invert so that suppression (lower power) maps to higher feature value
    total_power = np.mean(mus) + np.mean(betas)
    return 1.0 / (total_power + 1e-10)

def sliding_features(data, window_sec=2.0, stride_sec=0.5):
    win  = int(window_sec * sfreq)
    step = int(stride_sec * sfreq)
    feats, times = [], []
    t = 0
    while t + win <= data.shape[1]:
        feats.append(extract_motor_feature(data[:, t:t+win], sfreq))
        times.append(t / sfreq)
        t += step
    return np.array(feats), np.array(times)

print("Extracting mu/beta suppression features...")
feats_rest, _ = sliding_features(data_rest)
feats_task, _ = sliding_features(data_task)

all_feats  = np.concatenate([feats_rest, feats_task])
all_labels = np.concatenate([np.zeros(len(feats_rest)), np.ones(len(feats_task))])

# Normalize
all_feats_norm = (all_feats - all_feats.min()) / (all_feats.max() - all_feats.min() + 1e-10)
print(f"Total frames: {len(all_feats)} (rest: {len(feats_rest)}, task: {len(feats_task)})")

#  Kalman Filter 
class KalmanFilter1D:
    """
    1D Kalman Filter for latent motor state tracking.
    x_{t+1} = A * x_t + w_t,  w ~ N(0, Q)
    y_t     = C * x_t + v_t,  v ~ N(0, R)
    """
    def __init__(self, A=0.97, C=1.0, Q=0.005, R=0.03, x0=0.5, P0=0.5):
        self.A, self.C = A, C
        self.Q, self.R = Q, R
        self.x, self.P = x0, P0

    def step(self, y):
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q
        S = self.C * self.P * self.C + self.R
        K = self.P * self.C / S
        self.x = self.x + K * (y - self.C * self.x)
        self.P = (1 - K * self.C) * self.P
        return self.x, K, self.P

kf = KalmanFilter1D(x0=all_feats_norm[0])
states, gains, covs = [], [], []

print("Running Kalman filter...")
for y in all_feats_norm:
    x, K, P = kf.step(y)
    states.append(x)
    gains.append(K)
    covs.append(P)

states = np.array(states)
gains  = np.array(gains)
covs   = np.array(covs)

#  Quantitative Evaluation
threshold = np.median(states)
predicted_labels = (states > threshold).astype(int)
acc = accuracy_score(all_labels, predicted_labels)
auc = roc_auc_score(all_labels, states)

mean_rest = states[:len(feats_rest)].mean()
mean_task = states[len(feats_rest):].mean()
separation = mean_task - mean_rest

print("\n── Quantitative Results ─────────────────────────────")
print(f"Channels used           : {ch_used}")
print(f"Adaptive threshold      : {threshold:.3f}")
print(f"Classification accuracy : {acc*100:.1f}%")
print(f"AUC-ROC                 : {auc:.3f}")
print(f"Mean state (rest)       : {mean_rest:.3f}")
print(f"Mean state (task)       : {mean_task:.3f}")
print(f"State separation        : {separation:.3f}")
print(f"Final error covariance  : {covs[-1]:.6f}")

#  Noise Robustness 
noise_levels = [0.0, 0.05, 0.1, 0.2]
robustness_results = {}

for noise in noise_levels:
    kf_n = KalmanFilter1D(x0=all_feats_norm[0])
    noisy = all_feats_norm + np.random.normal(0, noise, len(all_feats_norm))
    est   = np.array([kf_n.step(y)[0] for y in noisy])
    acc_n = accuracy_score(all_labels, (est > threshold).astype(int))
    auc_n = roc_auc_score(all_labels, est)
    robustness_results[noise] = {'estimates': est, 'acc': acc_n, 'auc': auc_n}
    print(f"Noise σ={noise:.2f} → Acc: {acc_n*100:.1f}%  AUC: {auc_n:.3f}")

#  Plots 
fig, axes = plt.subplots(4, 1, figsize=(13, 12))
n_frames  = len(all_feats_norm)
frame_idx = np.arange(n_frames)
boundary  = len(feats_rest)

axes[0].plot(frame_idx, all_feats_norm, alpha=0.4, color='steelblue',
             linewidth=0.7, label='Raw EEG feature (μ/β suppression)')
axes[0].plot(frame_idx, states, color='crimson',
             linewidth=1.5, label='Kalman state estimate')
axes[0].axvline(boundary, color='black', linestyle='--', linewidth=1.2,
                label='Rest → Motor imagery transition')
axes[0].axhline(threshold, color='gray', linestyle=':', linewidth=1.0,
                label=f'Decision threshold ({threshold:.2f})')
axes[0].fill_between(frame_idx, 0, 1, where=(all_labels == 1),
                     alpha=0.08, color='orange', label='Motor imagery condition')
axes[0].set_title(
    f'Motor State Estimation (C3/Cz/C4)  |  Acc: {acc*100:.1f}%  AUC: {auc:.3f}')
axes[0].set_ylabel('State (norm.)')
axes[0].legend(fontsize=7)

axes[1].plot(frame_idx, gains, color='darkorange', linewidth=0.8)
axes[1].axvline(boundary, color='black', linestyle='--', linewidth=1.0)
axes[1].set_title('Kalman Gain K (observation trust)')
axes[1].set_ylabel('Gain K')

axes[2].plot(frame_idx, covs, color='seagreen', linewidth=0.8)
axes[2].axvline(boundary, color='black', linestyle='--', linewidth=1.0)
axes[2].set_title('Error Covariance P (estimation uncertainty)')
axes[2].set_ylabel('P')

for noise, res in robustness_results.items():
    axes[3].plot(frame_idx, res['estimates'], linewidth=0.9,
                 label=f'σ={noise:.2f}  Acc={res["acc"]*100:.0f}%  AUC={res["auc"]:.2f}',
                 alpha=0.8)
axes[3].axvline(boundary, color='black', linestyle='--', linewidth=1.0)
axes[3].set_title('Noise Robustness Study')
axes[3].set_ylabel('State Estimate')
axes[3].set_xlabel('Frame index')
axes[3].legend(fontsize=7)

plt.tight_layout()
# plt.savefig('cognitive_state_kalman_eegbci_v2.png', dpi=150)
plt.show()
print("\nPlot saved as cognitive_state_kalman_eegbci_v2.png")