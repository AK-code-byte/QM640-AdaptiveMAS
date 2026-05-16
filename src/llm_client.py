import os
from typing import Tuple, Dict

BACKBONE_MODEL_MAP = {
    'Claude-Sonnet':    'claude-sonnet-4-6',
    'Claude-Haiku':     'claude-haiku-4-5-20251001',
    'Llama-3.3-70B':    'meta-llama/Llama-3.3-70B-Instruct',
    'Ollama':           'ollama',
    'Gemini-2.0-Flash': 'gemini-2.0-flash',
}

# Set TEST_BACKEND in the notebook config cell to control which model runs during
# smoke tests and live API validation (max_rows=10 runs).
#   TEST_BACKEND = 'haiku'   → claude-haiku-4-5-20251001  (Anthropic)
#   TEST_BACKEND = 'ollama'  → local model served at http://localhost:11434/v1
TEST_BACKEND = os.getenv('TEST_BACKEND', 'haiku')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')

TEST_BACKBONE_MAP = {
    'haiku':  'Claude-Haiku',
    'ollama': 'Ollama',
}


def get_test_client() -> 'LLMClient':
    """Return a cheap LLMClient suitable for smoke tests and small validation runs."""
    backbone = TEST_BACKBONE_MAP.get(TEST_BACKEND, 'Claude-Haiku')
    return LLMClient(backbone=backbone)


class LLMClient:
    def __init__(self, backbone: str = 'GPT-4o-mini'):
        self.backbone = backbone
        self.model_id = BACKBONE_MODEL_MAP.get(backbone, backbone)
        self._init_client()

    def _init_client(self):
        if self.model_id == 'ollama':
            import openai
            self._client_type = 'openai'
            self._openai = openai.OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key='ollama',
            )
            self.model_id = OLLAMA_MODEL
        elif 'claude' in self.model_id.lower():
            import anthropic
            self._client_type = 'anthropic'
            self._anthropic = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY', '')
            )
        else:
            import openai
            self._client_type = 'openai'
            self._openai = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY', '')
            )

    def call(self, system_prompt: str, user_message: str) -> Tuple[str, Dict]:
        """
        Call the LLM with temperature=0 (hard requirement per architecture doc §6).
        Returns (response_text, {'prompt_tokens': int, 'completion_tokens': int}).
        """
        if self._client_type == 'anthropic':
            resp = self._anthropic.messages.create(
                model=self.model_id,
                max_tokens=512,
                temperature=0,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_message}],
            )
            if not resp.content:
                raise ValueError(f"Anthropic returned empty content for model {self.model_id}")
            text = resp.content[0].text
            usage = {
                'prompt_tokens': resp.usage.input_tokens,
                'completion_tokens': resp.usage.output_tokens,
            }
        else:
            resp = self._openai.chat.completions.create(
                model=self.model_id,
                temperature=0,
                max_tokens=512,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message},
                ],
            )
            if not resp.choices:
                raise ValueError(f"OpenAI returned empty choices for model {self.model_id}")
            text = resp.choices[0].message.content or ''
            usage = {
                'prompt_tokens': resp.usage.prompt_tokens,
                'completion_tokens': resp.usage.completion_tokens,
            }
        return text, usage


class AsyncLLMClient:
    """Async counterpart to LLMClient — uses AsyncAnthropic / async OpenAI.
    Required for asyncio.gather()-based config-parallel dev inference runs.
    """

    def __init__(self, backbone: str = 'Claude-Haiku'):
        self.backbone = backbone
        self.model_id = BACKBONE_MODEL_MAP.get(backbone, backbone)

    async def call(self, system_prompt: str, user_message: str):
        """Returns (response_text, usage_dict). Same signature as LLMClient.call."""
        if self.model_id == 'ollama' or self.backbone == 'Ollama':
            import asyncio
            loop = asyncio.get_event_loop()
            sync_client = LLMClient(backbone=self.backbone)
            return await loop.run_in_executor(None, sync_client.call, system_prompt, user_message)

        if 'claude' in self.model_id.lower():
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY', ''))
            resp = await client.messages.create(
                model=self.model_id,
                max_tokens=512,
                temperature=0,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_message}],
            )
            if not resp.content:
                raise ValueError(f'AsyncAnthropic returned empty content for {self.model_id}')
            return resp.content[0].text, {
                'prompt_tokens': resp.usage.input_tokens,
                'completion_tokens': resp.usage.output_tokens,
            }
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))
            resp = await client.chat.completions.create(
                model=self.model_id,
                temperature=0,
                max_tokens=512,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message},
                ],
            )
            if not resp.choices:
                raise ValueError(f'AsyncOpenAI returned empty choices for {self.model_id}')
            return resp.choices[0].message.content or '', {
                'prompt_tokens': resp.usage.prompt_tokens,
                'completion_tokens': resp.usage.completion_tokens,
            }
