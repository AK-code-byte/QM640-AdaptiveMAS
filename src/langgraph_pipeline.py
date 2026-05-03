from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from src.agents import (
    complexity_assessor, adaptive_router,
    run_single_agent, run_static_multi_agent,
    MedQAKnowledgeRetriever,
)
from src.token_tracker import TokenTracker


class AgentState(TypedDict):
    vignette: str
    ground_truth: str
    question_id: str
    complexity_score: float
    complexity_tier: str
    routing_decision: str
    predicted_answer: str
    total_tokens: int
    routing_overhead_tokens: int
    inference_tokens: int
    reasoning_steps: int


def build_routing_graph(client, retriever: Optional[MedQAKnowledgeRetriever] = None):
    """Build a LangGraph StateGraph implementing the 5-node routing architecture."""
    graph = StateGraph(AgentState)

    def assess_complexity(state: AgentState) -> AgentState:
        tier, score, usage = complexity_assessor(state['vignette'], client)
        t = TokenTracker()
        t.add(usage, is_routing=True)
        return {**state,
                'complexity_tier': tier,
                'complexity_score': score,
                'routing_overhead_tokens': t.routing_overhead}

    def route(state: AgentState) -> AgentState:
        routing = adaptive_router(state['complexity_score'])
        return {**state, 'routing_decision': routing}

    def run_solo(state: AgentState) -> AgentState:
        answer, tokens, steps = run_single_agent(state['vignette'], client)
        return {**state,
                'predicted_answer': answer,
                'inference_tokens': tokens,
                'total_tokens': state.get('routing_overhead_tokens', 0) + tokens,
                'reasoning_steps': steps}

    def run_group(state: AgentState) -> AgentState:
        tier = state['routing_decision']
        answer, tokens, steps = run_static_multi_agent(
            state['vignette'], client, tier=tier, retriever=retriever
        )
        return {**state,
                'predicted_answer': answer,
                'inference_tokens': tokens,
                'total_tokens': state.get('routing_overhead_tokens', 0) + tokens,
                'reasoning_steps': steps}

    graph.add_node('assess', assess_complexity)
    graph.add_node('route', route)
    graph.add_node('solo', run_solo)
    graph.add_node('group', run_group)

    graph.set_entry_point('assess')
    graph.add_edge('assess', 'route')
    graph.add_conditional_edges(
        'route',
        lambda s: 'solo' if s['routing_decision'] == 'solo' else 'group',
        {'solo': 'solo', 'group': 'group'}
    )
    graph.add_edge('solo', END)
    graph.add_edge('group', END)

    return graph.compile()
