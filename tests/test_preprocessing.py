# tests/test_preprocessing.py
import pandas as pd
import pytest
from src.preprocessing import (
    compute_tier_thresholds,
    compute_complexity_features,
    extract_features_single,
    NLP_FEATURE_COLS,
    load_medagentsbench_schema,
)


def _make_df(n=300):
    """Synthetic df with complexity_score spread across [0, 1]."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'clinical_vignette': ['Patient has fever and cough.'] * n,
        'dataset_source': ['MedQA'] * n,
        'complexity_score': rng.uniform(0, 1, n),
    })


def test_compute_tier_thresholds_splits_equally():
    df = _make_df(300)
    p33, p67 = compute_tier_thresholds(df)
    assert p33 < p67
    simple   = (df['complexity_score'] < p33).sum()
    moderate = ((df['complexity_score'] >= p33) & (df['complexity_score'] < p67)).sum()
    complex_ = (df['complexity_score'] >= p67).sum()
    for count in [simple, moderate, complex_]:
        assert abs(count / len(df) - 0.333) < 0.05


def test_compute_complexity_features_balanced_tiers():
    df = _make_df(300)
    result = compute_complexity_features(df)
    tier_pcts = result['task_complexity'].value_counts(normalize=True)
    for tier in ['Simple', 'Moderate', 'Complex']:
        assert abs(tier_pcts.get(tier, 0) - 0.333) < 0.05


def test_extract_features_single_no_has_structured_labs():
    feats = extract_features_single('Patient has fever.', 'MedQA')
    assert 'has_structured_labs' not in feats
    assert set(feats.keys()) == set(NLP_FEATURE_COLS)


def test_mimic_functions_removed():
    import src.preprocessing as pp
    assert not hasattr(pp, 'mimic_admission_to_vignette'), \
        'mimic_admission_to_vignette should be removed'
    assert not hasattr(pp, 'fit_complexity_calibration'), \
        'fit_complexity_calibration should be removed'
    assert not hasattr(pp, '_CALIBRATION_MODEL'), \
        '_CALIBRATION_MODEL global should be removed'


def test_medagentsbench_schema_validation():
    raw_list_format = [{'question': 'What is?', 'answer': 'X', 'type': 'diagnosis'}]
    records = load_medagentsbench_schema(raw_list_format)
    assert len(records) == 1
    assert records[0]['correct_answer'] == 'X'


def test_medagentsbench_nested_format():
    raw = {'data': [{'question': 'Q?', 'answer': 'A', 'type': 'diagnosis'}]}
    records = load_medagentsbench_schema(raw)
    assert len(records) == 1
    assert records[0]['correct_answer'] == 'A'
