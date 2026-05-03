from unittest.mock import MagicMock, patch
from src.llm_client import LLMClient, get_test_client, OLLAMA_BASE_URL

def _mock_openai_response(content='answer', prompt_tokens=10, completion_tokens=5):
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))],
        usage=MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )

def test_temperature_is_zero_for_openai():
    client = LLMClient(backbone='GPT-4o-mini')
    with patch.object(client._openai, 'chat') as mock_chat:
        mock_chat.completions.create.return_value = _mock_openai_response()
        client.call("system prompt", "user message")
        assert mock_chat.completions.create.call_args[1]['temperature'] == 0

def test_temperature_is_zero_for_haiku():
    client = LLMClient(backbone='Claude-Haiku')
    with patch.object(client._anthropic, 'messages') as mock_msg:
        mock_msg.create.return_value = MagicMock(
            content=[MagicMock(text='answer')],
            usage=MagicMock(input_tokens=10, output_tokens=5),
        )
        client.call("sys", "user")
        assert mock_msg.create.call_args[1]['temperature'] == 0

def test_backbone_model_map():
    c = LLMClient(backbone='GPT-4o')
    assert 'gpt-4o' in c.model_id
    c2 = LLMClient(backbone='Claude-Haiku')
    assert 'claude-haiku-4-5' in c2.model_id

def test_ollama_uses_local_base_url():
    client = LLMClient(backbone='Ollama')
    assert client._client_type == 'openai'
    assert OLLAMA_BASE_URL in str(client._openai.base_url)

def test_ollama_temperature_is_zero():
    client = LLMClient(backbone='Ollama')
    with patch.object(client._openai, 'chat') as mock_chat:
        mock_chat.completions.create.return_value = _mock_openai_response()
        client.call("sys", "user")
        assert mock_chat.completions.create.call_args[1]['temperature'] == 0

def test_returns_text_and_usage_dict():
    client = LLMClient(backbone='GPT-4o-mini')
    with patch.object(client._openai, 'chat') as mock_chat:
        mock_chat.completions.create.return_value = _mock_openai_response('diagnosis', 20, 8)
        text, usage = client.call("sys", "user")
        assert text == "diagnosis"
        assert usage == {'prompt_tokens': 20, 'completion_tokens': 8}

def test_get_test_client_returns_llm_client():
    import os
    os.environ['TEST_BACKEND'] = 'haiku'
    client = get_test_client()
    assert isinstance(client, LLMClient)
    assert 'haiku' in client.model_id.lower()


# ── AsyncLLMClient tests ──────────────────────────────────────────────────────

def test_async_client_instantiates_for_claude():
    from src.llm_client import AsyncLLMClient, BACKBONE_MODEL_MAP
    client = AsyncLLMClient(backbone='Claude-Haiku')
    assert client.model_id == BACKBONE_MODEL_MAP['Claude-Haiku']


def test_async_client_instantiates_for_gpt():
    from src.llm_client import AsyncLLMClient, BACKBONE_MODEL_MAP
    client = AsyncLLMClient(backbone='GPT-4o-mini')
    assert client.model_id == BACKBONE_MODEL_MAP['GPT-4o-mini']


def test_async_client_call_is_coroutine():
    from src.llm_client import AsyncLLMClient
    import inspect
    client = AsyncLLMClient(backbone='Claude-Haiku')
    assert inspect.iscoroutinefunction(client.call)
