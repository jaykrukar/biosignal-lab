"""Training pipeline deliberately preventing subject overlap across splits."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from .signals import SyntheticDataset
from .features import featurize, FEATURE_NAMES


@dataclass
class TrainedResult:
    report: dict[str, Any]
    actual: np.ndarray
    predicted: np.ndarray
    model: RandomForestClassifier


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
        "confusion_matrix_labels_0_1": confusion_matrix(y[test_idx], predictions, labels=[0, 1]).tolist(),
        "features": list(FEATURE_NAMES),
        "note": "Demo performance on synthetic data cannot establish clinical utility.",
    }
    return TrainedResult(report, y[test_idx], predictions, model)
