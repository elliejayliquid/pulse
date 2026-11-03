import os
import sys
from types import SimpleNamespace

import httpx
from openai import BadRequestError

sys.path.append(os.getcwd())

from core.llm import LLMClient


def _bad_request(raw: str) -> BadRequestError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {"message": "Provider returned error", "code": 400, "metadata": {"raw": raw}}
    return BadRequestError("Error code: 400 - Provider returned error", response=response, body=body)


class _FakeCompletions:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = []

    def create(self, **kwargs):
        # Snapshot: the client reuses (and mutates) its extra_body dict
        self.calls.append({**kwargs, "extra_body": dict(kwargs.get("extra_body") or {})})
        if self.errors:
            raise self.errors.pop(0)
        message = SimpleNamespace(content='{"action": "silent"}', tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def _client(errors) -> tuple[LLMClient, _FakeCompletions]:
    client = LLMClient("https://openrouter.ai/api/v1", "mistralai/mistral-large-4-0",
                       api_key="test", provider_type="openrouter", top_k=40)
    fake = _FakeCompletions(errors)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_reasoning_off_rejected_retries_without_it_and_remembers():
    client, fake = _client([_bad_request("reasoning_effort none is not supported for this model")])
    assert client.chat([{"role": "user", "content": "hi"}]) is not None
    assert fake.calls[0]["extra_body"]["reasoning"] == {"effort": "none"}
    assert "reasoning" not in fake.calls[1]["extra_body"]
    assert fake.calls[1]["extra_body"]["top_k"] == 40  # other extras survive

    client.chat([{"role": "user", "content": "again"}])
    assert len(fake.calls) == 3 and "reasoning" not in fake.calls[2]["extra_body"]


def test_unrelated_400_is_not_swallowed():
    client, fake = _client([_bad_request("context length exceeded")])
    assert client.chat([{"role": "user", "content": "hi"}]) is None
    assert len(fake.calls) == 1
    assert client._extra_body["reasoning"] == {"effort": "none"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
