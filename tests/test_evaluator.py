import pytest
from src.evaluator import answer_is_correct


def test_exact_match():
    assert answer_is_correct('hypertension', 'hypertension') == 1


def test_case_insensitive_match():
    assert answer_is_correct('Hypertension', 'hypertension') == 1


def test_fuzzy_match():
    assert answer_is_correct('type 2 diabetes mellitus', 'type 2 diabetes') == 1


def test_no_match():
    assert answer_is_correct('pneumonia', 'hypertension') == 0


def test_mimic_paths_removed():
    import src.evaluator as ev
    assert not hasattr(ev, 'ICD10_PATTERN'), 'ICD10_PATTERN should be removed'
    assert not hasattr(ev, 'ICD9_PATTERN'),  'ICD9_PATTERN should be removed'
    assert not hasattr(ev, '_normalise_icd'), '_normalise_icd should be removed'
    assert not hasattr(ev, 'ICD10_TO_ALIASES'), 'ICD10_TO_ALIASES should be removed'
    assert not hasattr(ev, 'ICD9_TO_ALIASES'),  'ICD9_TO_ALIASES should be removed'


def test_dataset_source_kwarg_accepted():
    # dataset_source kwarg should still be accepted for API compatibility
    result = answer_is_correct('pneumonia', 'pneumonia', dataset_source='MedQA')
    assert result == 1


def test_ddxplus_exact():
    assert answer_is_correct('acute laryngitis', 'acute laryngitis', dataset_source='DDXPlus') == 1


def test_ddxplus_fuzzy():
    assert answer_is_correct('acute laryngitis with pharyngitis', 'acute laryngitis',
                              dataset_source='DDXPlus') == 1
