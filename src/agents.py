from typing import Tuple, Optional
import numpy as np
from src.token_tracker import TokenTracker
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

AGENT_ROLES = {
    'solo':    ['Generalist'],
    'group-2': ['Internist', 'Specialist'],        # Specialist synthesizes; no separate Moderator
    'group-3': ['Internist', 'Specialist', 'Moderator'],
    'MDT':     ['Internist', 'Specialist', 'Radiologist', 'Pharmacist', 'Moderator'],
}

SYSTEM_PROMPTS = {
    'Generalist': (
        'You are a board-certified general-practice physician. '
        'Diagnose the following case concisely. '
        'Respond with the single most likely diagnosis only.'
    ),
    'Internist': (
        'You are a board-certified internist contributing to a multi-specialist '
        'diagnostic panel. Provide your primary diagnosis and key clinical reasoning.'
    ),
    'Specialist': (
        'You are a specialist (cardiology / pulmonology / neurology as appropriate). '
        'Critically evaluate the case and all prior opinions. '
        'State your final synthesized diagnosis.'
    ),
    'Radiologist': (
        'You are a radiologist. Comment on imaging findings implied by the case '
        'and their diagnostic implications.'
    ),
    'Pharmacist': (
        'You are a clinical pharmacist. Identify any drug-disease interactions '
        'or pharmacological clues relevant to the diagnosis.'
    ),
    'Moderator': (
        'You are the diagnostic moderator. Synthesize the inputs from your '
        'colleagues and return a single final diagnosis.'
    ),
}

COMPLEXITY_TIERS = ['Simple', 'Moderate', 'Complex']


def adaptive_router(complexity_score: float) -> str:
    """
    Maps a 0-1 complexity score to a routing tier using the architecture doc thresholds.
    <0.40 → solo, 0.40-0.60 → group-2, 0.60-0.80 → group-3, ≥0.80 → MDT
    """
    if complexity_score < 0.40:
        return 'solo'
    elif complexity_score < 0.60:
        return 'group-2'
    elif complexity_score < 0.80:
        return 'group-3'
    else:
        return 'MDT'


def complexity_assessor(vignette: str, client) -> Tuple[str, float, dict]:
    """Returns (complexity_tier_str, complexity_score_float, usage_dict)."""
    system = (
        'You are a medical triage classifier. '
        'Given a clinical vignette, return ONLY a JSON object in this exact format: '
        '{"complexity_score": <float 0-1>, "tier": "<Simple|Moderate|Complex>"}. '
        'No other text.'
    )
    import json, re
    result, usage = client.call(system, vignette)
    try:
        m = re.search(r'\{.*\}', result, re.DOTALL)
        if m is None:
            raise ValueError("No JSON object in response")
        parsed = json.loads(m.group())
        score = float(parsed.get('complexity_score', 0.5))
        tier = parsed.get('tier', 'Moderate')
        if tier not in COMPLEXITY_TIERS:
            tier = 'Moderate'
    except Exception:
        # Handle plain tier name (e.g. 'Simple') returned by mock or simplified LLM
        result_stripped = result.strip().capitalize()
        if result_stripped in COMPLEXITY_TIERS:
            tier = result_stripped
            score = {'Simple': 0.2, 'Moderate': 0.5, 'Complex': 0.8}[tier]
        else:
            score = 0.5
            tier = 'Moderate'
    score = max(0.0, min(1.0, score))
    return tier, score, usage


def run_single_agent(vignette: str, client) -> Tuple[str, int, int]:
    """Returns (answer, total_tokens, reasoning_steps=1)."""
    tracker = TokenTracker()
    answer, usage = client.call(SYSTEM_PROMPTS['Generalist'], vignette)
    tracker.add(usage, is_routing=False)
    return answer, tracker.total, 1


def run_static_multi_agent(vignette: str, client, tier: str = 'group-3',
                            retriever=None) -> Tuple[str, int, int]:
    """
    Static MAS with correct panel composition per tier.
    group-2: Internist deliberates, Specialist synthesizes (2 calls, no separate Moderator).
    group-3: Internist + Specialist deliberate, Moderator synthesizes (3 calls).
    MDT:     Internist + Specialist + Radiologist + Pharmacist deliberate,
             Moderator synthesizes with optional BM25 retrieval (5 calls).
    """
    tracker = TokenTracker()
    roles = AGENT_ROLES[tier]
    has_moderator = 'Moderator' in roles
    panel_roles = [r for r in roles if r != 'Moderator']
    opinions = []

    for role in panel_roles:
        ctx = vignette
        if opinions:
            ctx += '\n\nPrior opinions:\n' + '\n'.join(f'- {o}' for o in opinions)
        opinion, usage = client.call(SYSTEM_PROMPTS[role], ctx)
        tracker.add(usage, is_routing=False)
        opinions.append(f'{role}: {opinion}')

    if has_moderator:
        mod_ctx = vignette + '\n\nPanel inputs:\n' + '\n'.join(opinions)
        if tier == 'MDT' and retriever is not None:
            retrieved = retriever.retrieve(vignette, top_k=3)
            if retrieved:
                mod_ctx += '\n\nRelevant reference cases:\n' + retrieved
        final, usage = client.call(SYSTEM_PROMPTS['Moderator'], mod_ctx)
        tracker.add(usage, is_routing=False)
    else:
        # group-2: last panel agent (Specialist) provides final answer directly
        final = opinions[-1].split(': ', 1)[-1] if opinions else ''

    steps = len(panel_roles) + (1 if has_moderator else 0)
    return final, tracker.total, steps


def run_adaptive_multi_agent(
    vignette: str, client, retriever=None,
    precomputed_score: Optional[float] = None,
) -> Tuple[str, int, int, str, int]:
    """
    Adaptive routing. Returns (answer, total_tokens, steps, routing_tier, routing_overhead).
    Routing overhead (complexity_assessor tokens) tracked separately from inference tokens.
    Pass precomputed_score to skip the LLM-based complexity_assessor call when the NLP
    complexity score is already available in the dataset.
    """
    tracker = TokenTracker()

    if precomputed_score is not None:
        score = float(max(0.0, min(1.0, precomputed_score)))
        tracker.add({'prompt_tokens': 0, 'completion_tokens': 0}, is_routing=True)
    else:
        _, score, assess_usage = complexity_assessor(vignette, client)
        tracker.add(assess_usage, is_routing=True)

    routing = adaptive_router(score)

    if routing == 'solo':
        answer, inf_tokens, steps = run_single_agent(vignette, client)
    else:
        answer, inf_tokens, steps = run_static_multi_agent(
            vignette, client, tier=routing, retriever=retriever
        )

    tracker.add({'prompt_tokens': inf_tokens, 'completion_tokens': 0}, is_routing=False)

    return answer, tracker.total, steps, routing, tracker.routing_overhead


def safe_acc_per_1k(is_correct: int, token_cost: int) -> float:
    """Division-safe accuracy-per-1k-tokens. Returns NaN when token_cost == 0."""
    if token_cost <= 0:
        return float('nan')
    return is_correct / (token_cost / 1000)


class MedQAKnowledgeRetriever:
    """
    Lightweight BM25 index over MedQA question-answer pairs.
    Used exclusively by the MDT moderator to augment synthesis context.
    Architecture doc §3.1: BM25 retriever that produces the 11.8% accuracy improvement.
    """

    def __init__(self):
        self._index = None
        self._documents: list = []

    def build(self, df) -> None:
        """Build BM25 index from a DataFrame with 'clinical_vignette' and 'correct_answer' columns."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                'rank_bm25 is required for the MDT retriever. '
                'Install with: pip install rank_bm25'
            )
        docs = [
            f"{str(r.get('clinical_vignette', ''))} {str(r.get('correct_answer', ''))}"
            for _, r in df.iterrows()
        ]
        tokenised = [d.lower().split() for d in docs]
        self._index = BM25Okapi(tokenised)
        self._documents = docs
        print(f'BM25 index built: {len(docs):,} documents')

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return top-k matching passages as a formatted string. Returns '' if index not built."""
        if self._index is None or not self._documents:
            return ''
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        # Return top_k regardless of score — BM25 can return negative scores on small corpora.
        passages = [self._documents[i] for i in top_idx]
        if not passages:
            return ''
        return '\n'.join(
            f'[Ref {j + 1}] {p[:300]}' for j, p in enumerate(passages)
        )

    @property
    def is_ready(self) -> bool:
        return self._index is not None


def build_batch_request(custom_id: str, system_prompt: str, user_content: str,
                         model_id: str, max_tokens: int = 512):
    """
    Build a single Anthropic Batch API Request object.
    custom_id convention: '{question_id}__{role_tag}'
    e.g. 'MEDQA_001__sa', 'MEDQA_001__internist', 'MEDQA_001__specialist'
    """
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_content}],
        ),
    )


async def run_single_agent_async(vignette: str, client) -> Tuple[str, int, int]:
    """Async version of run_single_agent. Returns (answer, total_tokens, steps=1)."""
    tracker = TokenTracker()
    answer, usage = await client.call(SYSTEM_PROMPTS['Generalist'], vignette)
    tracker.add(usage, is_routing=False)
    return answer, tracker.total, 1


async def run_static_multi_agent_async(vignette: str, client,
                                        tier: str = 'group-3') -> Tuple[str, int, int]:
    """
    Async version of run_static_multi_agent.
    Each panel agent call is sequential (opinion chain — each sees prior opinions).
    Returns (answer, total_tokens, steps).
    """
    tracker = TokenTracker()
    roles = AGENT_ROLES[tier]
    has_moderator = 'Moderator' in roles
    panel_roles = [r for r in roles if r != 'Moderator']
    opinions = []

    for role in panel_roles:
        ctx = vignette
        if opinions:
            ctx += '\n\nPrior opinions:\n' + '\n'.join(f'- {o}' for o in opinions)
        opinion, usage = await client.call(SYSTEM_PROMPTS[role], ctx)
        tracker.add(usage, is_routing=False)
        opinions.append(f'{role}: {opinion}')

    if has_moderator:
        mod_ctx = vignette + '\n\nPanel inputs:\n' + '\n'.join(opinions)
        final, usage = await client.call(SYSTEM_PROMPTS['Moderator'], mod_ctx)
        tracker.add(usage, is_routing=False)
    else:
        final = opinions[-1].split(': ', 1)[-1] if opinions else ''

    steps = len(panel_roles) + (1 if has_moderator else 0)
    return final, tracker.total, steps
