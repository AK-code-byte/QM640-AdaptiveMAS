class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.routing_overhead = 0   # tokens used by complexity_assessor only
        self.inference_tokens = 0   # tokens used by actual agent calls

    def add(self, usage, is_routing: bool = False):
        if hasattr(usage, 'prompt_tokens'):
            p = usage.prompt_tokens
            c = usage.completion_tokens
        elif isinstance(usage, dict):
            p = usage.get('prompt_tokens', 0)
            c = usage.get('completion_tokens', 0)
        else:
            return
        self.prompt_tokens += p
        self.completion_tokens += c
        if is_routing:
            self.routing_overhead += p + c
        else:
            self.inference_tokens += p + c

    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens
