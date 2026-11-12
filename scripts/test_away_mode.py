"""Offline checks for heartbeat.away_mode and the /heartbeat command.

No model calls: the engine is built without __init__ and _do_heartbeat is faked.
Run: python scripts/test_away_mode.py
"""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engine import PulseEngine, TAPER_MAX_SECONDS  # noqa: E402
from core.llm import PulseResponse  # noqa: E402

failures = 0


def check(label, cond):
    global failures
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        failures += 1


def make_engine(mode="always", interval=1800):
    e = PulseEngine.__new__(PulseEngine)
    e.interval = interval
    e._heartbeat_randomize = False
    e._away_mode_config = mode
    e._away_mode_override = None
    e._unanswered_heartbeats = 0
    e._heartbeat_in_progress = False
    e._heartbeat_skipped = False
    e._last_activity = time.time()
    e._next_heartbeat_target = None
    e._last_tick_at = e._last_tick_action = e._last_tick_summary = None
    e.ran = []

    async def fake_do_heartbeat(manual=False, _e=e):
        _e.ran.append(manual)
        _e._heartbeat_skipped = getattr(_e, "skip_next", False)
        if _e._heartbeat_skipped:
            return None
        return PulseResponse(action=getattr(_e, "next_action", "silent"), message="hi")

    e._do_heartbeat = fake_do_heartbeat
    return e


def tick(e, manual=False):
    return asyncio.run(e.run_heartbeat_tick(manual=manual))


print("always")
e = make_engine("always")
for _ in range(3):
    tick(e)
check("interval never changes", e._away_interval() == 1800)
check("never waits for the human", not e.heartbeat_waiting_for_human())
check("counts unanswered anyway", e._unanswered_heartbeats == 3)

print("taper")
e = make_engine("taper")
waits = []
for _ in range(7):
    waits.append(e._away_interval())
    tick(e)
check("30m, 1h, 2h, 4h, 8h, 16h, then capped", waits == [1800, 3600, 7200, 14400, 28800, 57600, 57600])
e2 = make_engine("taper", interval=6 * 3600)
for _ in range(5):
    tick(e2)
check("never longer than 24h", e2._away_interval() == TAPER_MAX_SECONDS)
e._human_is_back()
check("human message resets taper", e._away_interval() == 1800)

print("once")
e = make_engine("once")
check("first heartbeat is due after the interval", e._away_interval() == 1800)
tick(e)
check("then waits for the human", e.heartbeat_waiting_for_human() and e._away_interval() is None)
check("no next-heartbeat target while waiting", e._next_heartbeat_target is None)
e._human_is_back()
e._mark_activity()
check("human message re-arms it", e._away_interval() == 1800 and e._next_heartbeat_target is not None)

print("skipped heartbeats don't count")
e = make_engine("once")
e.skip_next = True
tick(e)
check("quiet-hours/unreachable skip leaves count at 0", e._unanswered_heartbeats == 0)

print("manual /heartbeat")
e = make_engine("once")
tick(e)
check("waiting before manual", e.heartbeat_waiting_for_human())
tick(e, manual=True)
check("manual ran with manual=True", e.ran[-1] is True)
check("manual resets and isn't counted", e._unanswered_heartbeats == 0)
check("one automatic heartbeat may follow later", e._away_interval() == 1800)

print("override")
e = make_engine("always")
check("set once", e.set_away_mode("once") == "once" and e._away_mode_override == "once")
check("setting the config value clears the override", e.set_away_mode("always") == "always" and e._away_mode_override is None)
check("default clears", e.set_away_mode("taper") == "taper" and e.set_away_mode(None) == "always")
try:
    e.set_away_mode("sometimes")
    check("bad mode rejected", False)
except ValueError:
    check("bad mode rejected", True)

print("concurrency guard")
e = make_engine()
e._heartbeat_in_progress = True
try:
    tick(e)
    check("second tick refused", False)
except RuntimeError:
    check("second tick refused", True)

print("idle enrichment isn't starved while waiting")
e = make_engine("once")
tick(e)
e.db = None  # returns early either way; check the due logic directly
due = e._away_interval()
check("no heartbeat due while waiting", due is None)

print("telegram /heartbeat")
from channels.telegram import TelegramChannel  # noqa: E402


class FakeMessage:
    def __init__(self):
        self.replies = []
        self.message_id = 1

    async def reply_text(self, text, **kw):
        self.replies.append(text)


def run_cmd(engine, args):
    ch = TelegramChannel.__new__(TelegramChannel)
    ch._engine = engine

    class _T:
        def cancel(self):
            pass

    ch._start_typing_loop = lambda chat: _T()
    msg = FakeMessage()
    update = SimpleNamespace(message=msg, effective_chat=None)
    context = SimpleNamespace(args=args)
    asyncio.run(ch._cmd_heartbeat(update, context))
    return msg.replies


e = make_engine("always")
e.context = SimpleNamespace(ai_name="Kai")
r = run_cmd(e, ["once"])
check("/heartbeat once switches mode", e.away_mode == "once" and "until restart" in r[0])
r = run_cmd(e, ["mode"])
check("/heartbeat mode reports override", "once (until restart; config: always)" in r[0])
r = run_cmd(e, ["default"])
check("/heartbeat default restores config", e.away_mode == "always" and "from config" in r[0])
r = run_cmd(e, [])
check("/heartbeat runs a manual tick", e.ran[-1] is True)
check("silent result is reported", r and "quietly" in r[-1])
e.next_action = "notify"
r = run_cmd(e, [])
check("notify result sends no extra reply", r == [])
e.skip_next = True
r = run_cmd(e, [])
check("unreachable model is reported", r and "isn't reachable" in r[-1])
e.skip_next = False
r = run_cmd(e, ["sometimes"])
check("unknown word gets usage", r and "/heartbeat once" in r[0])

print("GUI config validation")
from gui.config_editor import ConfigEditor  # noqa: E402

ed = ConfigEditor.__new__(ConfigEditor)
check("away_mode once accepted", ed._clean_heartbeat_value("away_mode", "once") == "once")
try:
    ed._clean_heartbeat_value("away_mode", "never")
    check("bad away_mode rejected", False)
except ValueError:
    check("bad away_mode rejected", True)

print()
print("ALL PASSED" if not failures else f"{failures} FAILED")
sys.exit(1 if failures else 0)
