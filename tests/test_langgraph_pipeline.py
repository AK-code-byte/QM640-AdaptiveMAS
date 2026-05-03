from unittest.mock import MagicMock
from src.langgraph_pipeline import build_routing_graph, AgentState

def make_mock_client(responses):
    m = MagicMock()
    m.call.side_effect = responses
    return m

def test_state_carries_routing_decision():
    import json
    client = make_mock_client([
        (json.dumps({"complexity_score": 0.2, "tier": "Simple"}),
         {'prompt_tokens': 30, 'completion_tokens': 5}),
        ('flu diagnosis',
         {'prompt_tokens': 100, 'completion_tokens': 40}),
    ])
    graph = build_routing_graph(client)
    result = graph.invoke({
        'vignette': 'Patient presents with fever and sore throat.',
        'ground_truth': 'Influenza',
        'question_id': 'TEST_001',
    })
    assert result['routing_decision'] in ['solo', 'group-2', 'group-3', 'MDT']
    assert result['total_tokens'] > 0
    assert 'predicted_answer' in result

def test_graph_logs_routing_overhead_separately():
    import json
    client = make_mock_client([
        (json.dumps({"complexity_score": 0.9, "tier": "Complex"}),
         {'prompt_tokens': 30, 'completion_tokens': 5}),  # assessor
        ('internist diagnosis', {'prompt_tokens': 100, 'completion_tokens': 40}),
        ('specialist diagnosis', {'prompt_tokens': 110, 'completion_tokens': 45}),
        ('radiologist', {'prompt_tokens': 90, 'completion_tokens': 35}),
        ('pharmacist', {'prompt_tokens': 95, 'completion_tokens': 38}),
        ('final MDT answer', {'prompt_tokens': 200, 'completion_tokens': 80}),
    ])
    graph = build_routing_graph(client)
    result = graph.invoke({'vignette': 'Complex case.', 'ground_truth': 'X', 'question_id': 'T2'})
    assert result['routing_decision'] == 'MDT'
    assert result['routing_overhead_tokens'] == 35  # assessor only
