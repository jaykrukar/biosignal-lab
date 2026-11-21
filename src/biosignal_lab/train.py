"""Training pipeline deliberately preventing subject overlap across splits."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from .signals import SyntheticDataset
from .features import featurize, FEATURE_NAMES


@dataclass
class TrainedResult:
    report: dict[str, Any]
    actual: np.ndarray
    predicted: np.ndarray
    model: RandomForestClassifier


def _subject_bootstrap_intervals(actual: np.ndarray, predicted: np.ndarray,
                                 groups: np.ndarray, seed: int,
                                 replicates: int = 1000) -> dict[str, dict[str, float]]:
    """Percentile intervals from resampling held-out subjects as clusters."""
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(replicates)
    f1_samples = np.empty(replicates)
    for i in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled_groups])
        accuracy_samples[i] = accuracy_score(actual[indices], predicted[indices])
        f1_samples[i] = f1_score(actual[indices], predicted[indices], labels=[0, 1],
                                  average="macro", zero_division=0)

    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {"lower": round(float(low), 4), "upper": round(float(high), 4)}

    return {"accuracy": interval(accuracy_samples), "macro_f1": interval(f1_samples)}


def train_evaluate(dataset: SyntheticDataset, seed: int = 17) -> TrainedResult:
    X = featurize(dataset.waveforms, dataset.sample_rate)
    y = dataset.labels
    groups = dataset.subject_ids
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(split.split(X, y, groups=groups))
    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    assert train_subjects.isdisjoint(test_subjects), "SUBJECT LEAKAGE DETECTED"
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        raise ValueError("Insufficient class diversity; change dataset seed")
    model = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, n_jobs=1,
                                   random_state=seed)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    probabilities = model.predict_proba(X[test_idx])[:, list(model.classes_).index(1)]
    test_groups = groups[test_idx]
    report = {
        "dataset": "synthetic pulse simulation (NOT actual medical data)",
        "seed": seed,
        "n_segments": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_subject_count": len(train_subjects),
        "test_subject_count": len(test_subjects),
        "subject_overlap_count": len(train_subjects & test_subjects),
        "accuracy": round(float(accuracy_score(y[test_idx], predictions)), 4),
        "macro_f1": round(float(f1_score(y[test_idx], predictions, average="macro")), 4),
        "brier_score": round(float(brier_score_loss(y[test_idx], probabilities)), 4),
        "subject_cluster_bootstrap_95_percent_ci": _subject_bootstrap_intervals(
            y[test_idx], predictions, test_groups, seed
        ),
        "bootstrap_resampling_unit": "held-out subject (all that subject's segments travel together)",
        "bootstrap_replicates": 1000,
        "confusion_matrix_labels_0_1": confusion_matrix(y[test_idx], predictions, labels=[0, 1]).tolist(),
        "features": list(FEATURE_NAMES),
        "note": "Synthetic demonstration only. Subject-cluster bootstrap intervals use a small held-out cohort and are illustrative, not clinical evidence.",
    }
    return TrainedResult(report, y[test_idx], predictions, model)
