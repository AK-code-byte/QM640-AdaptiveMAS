import re
from typing import Optional
from rapidfuzz import fuzz


def _clean(s: str) -> str:
    return re.sub(r'[^\w\s]', '', str(s).lower().strip())


def answer_is_correct(
    predicted: str,
    ground_truth: str,
    dataset_source: Optional[str] = None,
    fuzzy_threshold: int = 88,
) -> int:
    """
    Correctness evaluation: exact match after normalisation, then rapidfuzz token-set ratio.
    dataset_source parameter retained for API compatibility but does not alter logic.
    """
    p_clean = _clean(predicted)
    g_clean = _clean(ground_truth)
    if p_clean == g_clean:
        return 1
    return int(fuzz.token_set_ratio(p_clean, g_clean) >= fuzzy_threshold)
