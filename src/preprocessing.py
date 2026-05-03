import json
import re
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def load_medagentsbench_schema(raw) -> List[Dict]:
    """
    Parse MedAgentsBench data handling both flat list and nested dict formats.
    Validates that no question or answer field is silently empty.
    """
    if isinstance(raw, dict):
        items = raw.get('data', raw.get('tasks', list(raw.values())[0] if raw else []))
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f'Unexpected MedAgentsBench format: {type(raw)}')

    records = []
    for i, item in enumerate(items):
        question = item.get('question') or item.get('input') or ''
        answer = item.get('answer') or item.get('output') or ''
        if not question or not answer:
            raise ValueError(
                f'MedAgentsBench item {i} has empty question or answer. '
                f'Check schema. Keys present: {list(item.keys())}'
            )
        records.append({
            'question_id':       f'MAGBENCH_{i}',
            'dataset_source':    'MedAgentsBench',
            'clinical_vignette': question,
            'correct_answer':    str(answer),
        })
    return records


# ── Complexity feature extraction ─────────────────────────────────────────────

_SPACY_MODEL = None
_SYMPTOM_PATTERN = re.compile(
    r'\b(fever|pain|dyspnea|cough|fatigue|nausea|vomiting|dizziness|headache|'
    r'rash|edema|tachycardia|bradycardia|hypotension|hypertension|confusion|'
    r'weakness|bleeding|syncope|palpitations|dysphagia|pruritus|jaundice|'
    r'hemoptysis|hematuria|polyuria|polydipsia|anorexia)\b',
    re.IGNORECASE
)


def _get_spacy():
    """Lazy-load scispaCy model; falls back to regex if unavailable.

    scispaCy requires Python <=3.12. On Python 3.13+ the import itself will
    fail with ImportError, which is caught here so the regex path activates
    automatically without any user action.
    """
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            import spacy
            _SPACY_MODEL = spacy.load('en_core_sci_lg')
        except (ImportError, OSError):
            _SPACY_MODEL = 'fallback'
    return _SPACY_MODEL


def extract_features_single(text: str, dataset_source: str = '') -> dict:
    """
    Extract the 4-dimensional feature vector per architecture doc §1.2.
    Returns: num_symptoms, num_conditions, clinical_vignette_length, step_level.
    """
    text_str = str(text)
    nlp = _get_spacy()
    if nlp != 'fallback':
        doc = nlp(text_str[:5000])
        n_symp = len({
            e.text.lower() for e in doc.ents
            if e.label_ in ('SIGN_OR_SYMPTOM', 'FINDING')
        }) or len(set(_SYMPTOM_PATTERN.findall(text_str)))
        n_cond = max(len({
            e.text.lower() for e in doc.ents
            if e.label_ in ('DISEASE', 'CONDITION')
        }), 1)
    else:
        n_symp = len(set(_SYMPTOM_PATTERN.findall(text_str)))
        n_cond = 1

    v_len = len(text_str.split())
    step_lv = 2 if 'medqa' in str(dataset_source).lower() else 0
    return {
        'num_symptoms': n_symp,
        'num_conditions': n_cond,
        'clinical_vignette_length': v_len,
        'step_level': step_lv,
    }


NLP_FEATURE_COLS = [
    'num_symptoms', 'num_conditions', 'clinical_vignette_length', 'step_level',
]


def _compute_heuristic_score(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic complexity score in [0, 1].
    score = (vlen/400)*0.4 + (nsymp/15)*0.4 + (ncond/10)*0.2
    Fixed normalisation constants prevent test-set leakage.
    """
    if 'clinical_vignette_length' in df.columns:
        vlen = df['clinical_vignette_length']
    elif 'clinical_vignette' in df.columns:
        vlen = df['clinical_vignette'].apply(lambda t: len(str(t).split()))
    else:
        vlen = pd.Series(0, index=df.index)

    nsymp = df['num_symptoms']   if 'num_symptoms'   in df.columns else pd.Series(0, index=df.index)
    ncond = df['num_conditions'] if 'num_conditions' in df.columns else pd.Series(1, index=df.index)

    return (
        vlen.fillna(0).clip(upper=400)  / 400  * 0.4
        + nsymp.fillna(0).clip(upper=15) / 15  * 0.4
        + ncond.fillna(1).clip(upper=10) / 10  * 0.2
    ).clip(0.0, 1.0).round(4)


def compute_tier_thresholds(df: pd.DataFrame) -> tuple:
    """
    Return (p33, p67) — the 33rd and 67th percentile of complexity_score in df.
    Using these as tier boundaries produces ~equal Simple/Moderate/Complex counts.
    Must be called after complexity_score column exists on df.
    """
    if 'complexity_score' not in df.columns:
        raise ValueError('complexity_score column must exist before calling compute_tier_thresholds()')
    p33 = float(df['complexity_score'].quantile(0.333))
    p67 = float(df['complexity_score'].quantile(0.667))
    return p33, p67


def compute_complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end complexity feature pipeline.
    Stage 1: extract 5D NLP features (scispaCy NER or regex fallback).
    Stage 2: compute calibrated complexity_score (APR-DRG logistic regression or heuristic).
    Stage 3: derive task_complexity categorical from score thresholds.
    """
    from tqdm.auto import tqdm
    df = df.copy()

    needs_extraction = (
        'clinical_vignette' in df.columns and
        any(c not in df.columns or df[c].isna().any() for c in NLP_FEATURE_COLS)
    )
    if needs_extraction:
        feat_rows = [
            extract_features_single(
                r['clinical_vignette'],
                dataset_source=r.get('dataset_source', '')
            )
            for _, r in tqdm(df.iterrows(), total=len(df), desc='Feature extraction')
        ]
        feat_df = pd.DataFrame(feat_rows, index=df.index)
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col]
            elif df[col].isna().any():
                # Preserve structured values (e.g. DDXPlus num_symptoms); fill NaN rows only
                df[col] = df[col].fillna(feat_df[col])

    if 'complexity_score' not in df.columns:
        df['complexity_score'] = _compute_heuristic_score(df)

    if 'task_complexity' not in df.columns:
        p33, p67 = compute_tier_thresholds(df)
        df['task_complexity'] = pd.cut(
            df['complexity_score'],
            bins=[-0.01, p33, p67, 1.01],
            labels=['Simple', 'Moderate', 'Complex'],
            right=False,
        ).astype(str)

    return df
