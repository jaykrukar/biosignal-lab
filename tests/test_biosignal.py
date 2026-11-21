import numpy as np
import pytest
from biosignal_lab.signals import make_synthetic_dataset
from biosignal_lab.features import extract_features, featurize
from biosignal_lab.train import train_evaluate


def test_generation_deterministic():
    a = make_synthetic_dataset(seed=11, n_subjects=8)
    b = make_synthetic_dataset(seed=11, n_subjects=8)
    assert np.array_equal(a.waveforms, b.waveforms)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.subject_ids, b.subject_ids)


def test_generate_correct_shapes():
    ds = make_synthetic_dataset(n_subjects=5, samples_per_subject=6)
    assert ds.waveforms.shape == (30, 400)
    assert len(np.unique(ds.subject_ids)) == 5
    assert set(ds.labels) == {0, 1}


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        make_synthetic_dataset(n_subjects=1)
    with pytest.raises(ValueError):
        extract_features(np.array([np.nan]*30), 100)
    with pytest.raises(ValueError):
        extract_features(np.arange(30), 0)


def test_features_are_finite():
    ds = make_synthetic_dataset(n_subjects=6, samples_per_subject=4)
    values = featurize(ds.waveforms, ds.sample_rate)
    assert values.shape == (24, 8)
    assert np.isfinite(values).all()


def test_no_group_leakage_and_repeatable_metrics():
    ds = make_synthetic_dataset()
    first = train_evaluate(ds)
    second = train_evaluate(ds)
    assert first.report == second.report
    assert first.report["subject_overlap_count"] == 0
    assert first.report["n_train"] + first.report["n_test"] == 360
    assert 0 <= first.report["macro_f1"] <= 1
    assert 0 <= first.report["brier_score"] <= 1
    assert first.report["subject_cluster_bootstrap_95_percent_ci"] == second.report[
        "subject_cluster_bootstrap_95_percent_ci"
    ]
    assert first.report["bootstrap_replicates"] == 1000
    for interval in first.report["subject_cluster_bootstrap_95_percent_ci"].values():
        assert 0 <= interval["lower"] <= interval["upper"] <= 1
