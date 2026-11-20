"""Small deterministic feature extraction layer with input validation."""
import numpy as np
from scipy.signal import find_peaks

FEATURE_NAMES = ("mean", "std", "rms", "mean_abs_diff", "peak_count",
                 "rr_mean", "rr_std", "spectral_high_ratio")


def extract_features(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1 or wave.size < 20 or not np.all(np.isfinite(wave)):
        raise ValueError("wave must be a finite one-dimensional array with >=20 samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    minimum_separation = max(1, int(sample_rate * .35))
    threshold = np.median(wave) + max(.15, .35 * np.std(wave))
    peaks, _ = find_peaks(wave, height=threshold, distance=minimum_separation)
    rr = np.diff(peaks) / sample_rate
    freqs = np.fft.rfftfreq(wave.size, d=1 / sample_rate)
    power = np.abs(np.fft.rfft(wave - np.mean(wave))) ** 2
    high_ratio = float(power[freqs >= 5].sum() / (power.sum() + 1e-12))
    values = np.array([
        np.mean(wave), np.std(wave), np.sqrt(np.mean(wave ** 2)),
        np.mean(np.abs(np.diff(wave))), len(peaks),
        np.mean(rr) if rr.size else 0.0,
        np.std(rr) if rr.size else 0.0,
        high_ratio,
    ], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite feature generated")
    return values


def featurize(waveforms: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.vstack([extract_features(wave, sample_rate) for wave in waveforms])
