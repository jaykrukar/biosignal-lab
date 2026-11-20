"""Generate explicitly synthetic pulse waveforms for software testing.

These signals are NOT clinical ECG/EEG data. They are deliberately simplified.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    waveforms: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    sample_rate: int


def make_synthetic_dataset(n_subjects: int = 30, samples_per_subject: int = 12,
                           sample_rate: int = 100, duration: float = 4.0,
                           seed: int = 17) -> SyntheticDataset:
    if n_subjects < 5 or samples_per_subject < 4 or sample_rate < 20 or duration < 2:
        raise ValueError("Need >=5 subjects, >=4 samples/subject, >=20 Hz, >=2 sec")
    rng = np.random.default_rng(seed)
    n_samples = round(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    records, labels, subject_ids = [], [], []
    for subject in range(n_subjects):
        baseline = rng.normal(0.0, 0.03)
        amplitude = rng.uniform(0.8, 1.2)
        for _ in range(samples_per_subject):
            label = int(rng.integers(0, 2))
            # class 0: relatively regular pulses. class 1: irregular spacing.
            rate = rng.uniform(0.74, 0.92)
            pulse_times = []
            current = 0.28
            while current < duration:
                pulse_times.append(current)
                jitter = rng.normal(0, 0.13) if label else rng.normal(0, 0.015)
                current += max(0.40, rate + jitter)
            signal = np.full(n_samples, baseline)
            for peak in pulse_times:
                signal += amplitude * np.exp(-0.5 * ((t - peak) / 0.040) ** 2)
            signal += 0.045 * np.sin(2 * np.pi * 0.45 * t)
            signal += rng.normal(0.0, 0.018, size=n_samples)
            records.append(signal)
            labels.append(label)
            subject_ids.append(f"subject-{subject:03d}")
    return SyntheticDataset(np.asarray(records), np.asarray(labels, dtype=int),
                            np.asarray(subject_ids), sample_rate)
