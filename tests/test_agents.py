from unittest.mock import patch, MagicMock
from src.agents import adaptive_router, run_static_multi_agent, run_adaptive_multi_agent

def make_llm(text='diagnosis', prompt_tokens=100, completion_tokens=50):
    m = MagicMock()
    m.call.return_value = (text, {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})
    return m

def test_adaptive_router_all_four_tiers():
    assert adaptive_router(0.20) == 'solo'
    assert adaptive_router(0.50) == 'group-2'
    assert adaptive_router(0.70) == 'group-3'
    assert adaptive_router(0.90) == 'MDT'

def test_adaptive_router_boundary_values():
    assert adaptive_router(0.40) == 'group-2'
    assert adaptive_router(0.60) == 'group-3'
    assert adaptive_router(0.80) == 'MDT'

def test_group2_uses_internist_and_specialist_no_separate_moderator():
    client = make_llm()
    ans, tokens, steps = run_static_multi_agent('vignette', client, tier='group-2')
    assert client.call.call_count == 2
    assert steps == 2

def test_group3_uses_three_agents_with_moderator():
    client = make_llm()
    ans, tokens, steps = run_static_multi_agent('vignette', client, tier='group-3')
    assert client.call.call_count == 3
    assert steps == 3

def test_mdt_uses_four_specialist_plus_moderator():
    client = make_llm()
    ans, tokens, steps = run_static_multi_agent('vignette', client, tier='MDT')
    assert client.call.call_count == 5

def test_token_tracking_captures_all_agents_in_adaptive():
    client = make_llm(prompt_tokens=100, completion_tokens=50)
    assessor_response = ('Simple', {'prompt_tokens': 30, 'completion_tokens': 5})
    solo_response = ('diagnosis', {'prompt_tokens': 100, 'completion_tokens': 50})
    client.call.side_effect = [assessor_response, solo_response]
    ans, total_tokens, steps, routing, overhead = run_adaptive_multi_agent('vignette', client)
    assert overhead == 35
    assert total_tokens == 35 + 150
    assert routing == 'solo'

def test_acc_per_1k_tokens_no_divide_by_zero():
    import math
    from src.agents import safe_acc_per_1k
    assert math.isnan(safe_acc_per_1k(1, 0))
    assert safe_acc_per_1k(1, 1000) == 1.0


# Task 5a: BM25 retriever tests

def test_bm25_retriever_build_and_retrieve():
    import pandas as pd
    from src.agents import MedQAKnowledgeRetriever
    df = pd.DataFrame([
        {'clinical_vignette': 'Patient with fever and cough', 'correct_answer': 'pneumonia'},
        {'clinical_vignette': 'Patient with chest pain and dyspnea', 'correct_answer': 'NSTEMI'},
        {'clinical_vignette': 'Elderly patient with confusion', 'correct_answer': 'delirium'},
    ])
    ret = MedQAKnowledgeRetriever()
    ret.build(df)
    result = ret.retrieve('fever cough productive sputum', top_k=1)
    assert 'pneumonia' in result.lower() or '[Ref' in result

def test_bm25_retriever_empty_returns_empty_string():
    from src.agents import MedQAKnowledgeRetriever
    ret = MedQAKnowledgeRetriever()
    assert ret.retrieve('any query') == ''

def test_mdt_moderator_receives_retrieved_passages():
    import pandas as pd
    from src.agents import MedQAKnowledgeRetriever
    client = make_llm()
    df = pd.DataFrame([
        {'clinical_vignette': 'patient with hypertension', 'correct_answer': 'hypertension'},
    ])
    retriever = MedQAKnowledgeRetriever()
    retriever.build(df)
    ans, tokens, steps = run_static_multi_agent(
        'patient with high blood pressure', client, tier='MDT', retriever=retriever
    )
    assert client.call.call_count == 5
    moderator_call_args = client.call.call_args_list[-1]
    moderator_user_prompt = moderator_call_args[0][1]
    assert 'Relevant reference cases' in moderator_user_prompt or 'hypertension' in moderator_user_prompt
