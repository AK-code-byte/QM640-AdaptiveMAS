from src.token_tracker import TokenTracker

def test_add_dict_usage():
    t = TokenTracker()
    t.add({'prompt_tokens': 100, 'completion_tokens': 50})
    assert t.total == 150

def test_add_object_usage():
    class FakeUsage:
        prompt_tokens = 200
        completion_tokens = 80
    t = TokenTracker()
    t.add(FakeUsage())
    assert t.total == 280

def test_routing_overhead_tracked_separately():
    t = TokenTracker()
    t.add({'prompt_tokens': 50, 'completion_tokens': 10}, is_routing=True)
    t.add({'prompt_tokens': 300, 'completion_tokens': 100}, is_routing=False)
    assert t.routing_overhead == 60
    assert t.inference_tokens == 400
    assert t.total == 460

def test_solo_routing_adds_correctly():
    t = TokenTracker()
    t.add({'prompt_tokens': 500, 'completion_tokens': 0})
    assert t.total == 500
    assert t.prompt_tokens == 500
