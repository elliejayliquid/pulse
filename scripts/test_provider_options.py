"""Tests for the provider options: unreadable heartbeat replies, the OpenAI
cached-token field, and provider.summarization_model on each provider.

Run from the Pulse folder:  python scripts/test_provider_options.py
No API calls are made.
"""
import asyncio
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.append(os.getcwd())

import yaml

from core.db import PulseDatabase
from core.engine import PulseEngine
from core.llm import PulseResponse
from core.usage import UsageTracker


def test_unreadable_reply_sends_nothing_and_is_flagged():
    bad = PulseResponse.from_llm_output("Hmm, I think I'll say hi... actually let me reconsider")
    assert bad.action == "silent" and bad.message == "" and bad.parse_failed
    good = PulseResponse.from_llm_output('{"thinking": "x", "action": "notify", "message": "hi"}')
    assert good.action == "notify" and good.message == "hi" and not good.parse_failed


def test_parse_failure_is_reported_and_notice_is_rate_limited():
    with tempfile.TemporaryDirectory() as tmp:
        e = PulseEngine.__new__(PulseEngine)
        e.config = {"_persona_dir": tmp, "_persona_name": "ada"}
        e.context = SimpleNamespace(ai_name="Ada")
        e._last_parse_notice = 0.0
        actions, sent = [], []
        e._log_action = lambda action, tools_used=None, summary="": actions.append((action, summary))

        async def fake_dispatch(resp):
            sent.append(resp)
        e._dispatch = fake_dispatch

        asyncio.run(e._report_parse_failure("heartbeat", ["web_search"]))
        asyncio.run(e._report_parse_failure("task"))

        lines = (Path(tmp) / "data" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(l)["tool"] for l in lines] == ["<heartbeat: parse>", "<task: parse>"]
        assert [a for a, _ in actions] == ["parse_error", "parse_error"]
        assert len(sent) == 1, "second failure within the hour must not send another notice"
        assert sent[0].action == "notify"
        assert sent[0].message.startswith("⚠️ HB-PARSE: Ada's heartbeat reply couldn't be read")


def test_openai_cached_tokens_have_their_own_field():
    with tempfile.TemporaryDirectory() as tmp:
        db = PulseDatabase(Path(tmp) / "pulse.db")
        tracker = UsageTracker({"_db": db})
        tracker.record(prompt_tokens=1000, completion_tokens=50, model="gpt", cached_prompt_tokens=600)
        tracker.record(prompt_tokens=200, completion_tokens=10, model="claude",
                       cache_read_input_tokens=300)
        today = tracker.get_today()
        assert today["prompt_tokens"] == 1200
        assert today["cached_prompt_tokens"] == 600
        assert today["cache_read_input_tokens"] == 300
        db.conn.close()

        json_tracker = UsageTracker({"paths": {"usage": str(Path(tmp) / "usage.json")}})
        json_tracker.record(prompt_tokens=10, completion_tokens=1, cached_prompt_tokens=4)
        assert json_tracker.get_today()["cached_prompt_tokens"] == 4
        assert json_tracker.get_today()["cache_read_input_tokens"] == 0


def _engine(provider: dict):
    with open("config.yaml", encoding="utf-8") as f:
        config = copy.deepcopy(yaml.safe_load(f))
    config["provider"].update(provider)
    return PulseEngine(config, channels={}, llm_endpoint="http://127.0.0.1:9/v1", api_key="test")


def test_summarization_model_on_each_provider():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(cwd)
        os.chdir(tmp)
        (Path(tmp) / "config.yaml").write_text((root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        try:
            for ptype in ("openrouter", "custom", "openai", "anthropic"):
                e = _engine({"type": ptype, "model": "big-model", "summarization_model": "small-model"})
                assert e.summarizer_llm is not e.llm, ptype
                assert type(e.summarizer_llm) is type(e.llm), ptype
                assert e.summarizer_llm.model_name == "small-model" and e.llm.model_name == "big-model", ptype

            e = _engine({"type": "local", "summarization_model": "small-model"})
            assert e.summarizer_llm is e.llm, "local llama-server: ignored, persona's model is used"

            e = _engine({"type": "openrouter", "model": "big-model", "summarization_model": ""})
            assert e.summarizer_llm is e.llm
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
