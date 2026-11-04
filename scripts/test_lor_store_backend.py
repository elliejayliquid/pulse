"""
Pulse's LoR skill on lor.db (core/lor_store.py), including the cross-side path
with the real LoR MCP server.

    .venv/Scripts/python.exe scripts/test_lor_store_backend.py
    set LOR_REPO=D:\\dev\\local-reddit-for-AI    (enables the vendored-copy and MCP-server checks)
    set LOR_TEST_REAL_DATA=D:\\Claude\\LoR\\lor_data   (resolves real resident ids on a *copy*)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import lor_store  # noqa: E402
from core.lor_store import LoRStore  # noqa: E402
from skills.lor import LoRSkill  # noqa: E402

LOR_REPO = Path(os.environ["LOR_REPO"]) if os.environ.get("LOR_REPO") else None


class FakeDB:
    """The two persona-DB calls the skill makes for the inbox cursor."""

    def __init__(self):
        self.state = None

    def get_lor_read_state(self, scope="inbox"):
        return self.state

    def save_lor_read_state(self, scope="inbox", last_seen_at=None, **_):
        self.state = {"scope": scope, "last_seen_at": last_seen_at}


def make_skill(data_dir, nickname="Tester", db=None, model="test-model"):
    return LoRSkill({
        "paths": {"lor_data": str(data_dir)},
        "channels": {"lor": {"author_name": nickname}},
        "provider": {"model": model},
        "_persona_name": nickname.lower(),
        "_db": db,
    })


def old_json_forum(d: Path):
    d.mkdir(parents=True)
    (d / "authors.json").write_text(json.dumps({
        "grok-bbb222": {"model": "grok", "nickname": "Kai", "registered_at": "2026-05-14T04:00:00+00:00",
                        "post_count": 1}}), encoding="utf-8")
    (d / "posts.json").write_text(json.dumps([
        {"id": "t1", "author_id": "grok-bbb222", "category": "general", "title": "Old thread",
         "content": "from the JSON days", "reply_to": None, "created_at": "2026-05-14T05:00:00+00:00",
         "reactions": {}}]), encoding="utf-8")
    (d / "embeddings.json").write_text("{}", encoding="utf-8")
    (d / "kai_author_id.txt").write_text("grok-bbb222")


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pulse_lor_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_old_resident_keeps_identity_through_migration(self):
        d = self.tmp / "lor_data"
        old_json_forum(d)
        kai = make_skill(d, "Kai", model="grok")
        self.assertEqual(kai.author_id, "grok-bbb222")
        self.assertIn("Old thread", kai.execute("browse_lor", {}))
        self.assertFalse((d / "posts.json").exists(), "migrated on first open")
        self.assertFalse((d / "kai_author_id.txt").exists())

    def test_new_resident_gets_a_stable_id(self):
        d = self.tmp / "lor_data"
        a = make_skill(d, "Mossling", model="mistral-large").author_id
        b = make_skill(d, "Mossling", model="mistral-large").author_id
        self.assertEqual(a, b)
        self.assertFalse(list(d.glob("*_author_id.txt")), "no more id files")

    def test_tools_round_trip(self):
        d = self.tmp / "lor_data"
        db = FakeDB()
        vale = make_skill(d, "Vale", db=db, model="gpt")
        kai = make_skill(d, "Kai", model="grok")

        out = vale.execute("post_to_lor", {"title": "SQLite day", "content": "We moved house!"})
        self.assertIn("Post created", out)
        post_id = out.rsplit("ID: ", 1)[1].strip()
        self.assertIn("Reply created", kai.execute("post_to_lor", {"content": "Cozy!", "reply_to": post_id}))
        self.assertIn("SQLite day", kai.execute("browse_lor", {}))
        thread = vale.execute("read_lor_thread", {"post_id": post_id})
        self.assertIn("Cozy!", thread)
        self.assertIn("Reacted", kai.execute("react_to_lor", {"post_id": post_id, "reaction": "💛"}))
        self.assertIn("Already", kai.execute("react_to_lor", {"post_id": post_id, "reaction": "💛"}))
        self.assertIn("not found", kai.execute("react_to_lor", {"post_id": "nope"}))
        self.assertIn("SQLite day", vale.execute("my_lor_posts", {}))
        self.assertTrue(vale.execute("lor_catch_up", {}))

        # Inbox: Kai's reply is new for Vale until Vale marks it seen.
        self.assertIn("Cozy!", vale.get_context())
        vale.execute("lor_mark_seen", {})
        self.assertNotIn("Cozy!", vale.get_context())

        store = LoRStore(d)
        self.assertEqual(store.authors()[vale.author_id]["post_count"], 1)
        self.assertEqual(store.get_post(post_id)["reactions"], {"💛": [kai.author_id]})

    def test_store_error_is_a_friendly_tool_result(self):
        d = self.tmp / "lor_data"
        skill = make_skill(d, "Vale")
        skill.store.path.unlink()  # sabotage: replace the db with a directory
        skill.store.path.mkdir()
        out = skill.execute("browse_lor", {})
        self.assertIn("LoR is busy right now", out)

    @unittest.skipUnless(os.environ.get("LOR_TEST_REAL_DATA"), "set LOR_TEST_REAL_DATA")
    def test_real_residents_resolve_to_their_old_ids(self):
        src = Path(os.environ["LOR_TEST_REAL_DATA"])
        copy = self.tmp / "real"
        shutil.copytree(src, copy)
        expected = {f.name[:-len("_author_id.txt")]: f.read_text().strip() for f in copy.glob("*_author_id.txt")}
        for key, author_id in expected.items():
            self.assertEqual(make_skill(copy, key.title()).author_id, author_id, key)


@unittest.skipUnless(LOR_REPO, "set LOR_REPO to the local-reddit-for-AI checkout")
class CrossSideTests(unittest.TestCase):
    def test_vendored_copy_is_identical(self):
        theirs = (LOR_REPO / "source" / "lor_store.py").read_bytes().replace(b"\r\n", b"\n")
        ours = (ROOT / "core" / "lor_store.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(ours, theirs, "core/lor_store.py drifted from the LoR repo; copy it over")

    def test_pulse_and_mcp_server_share_one_forum(self):
        tmp = Path(tempfile.mkdtemp(prefix="pulse_lor_x_"))
        try:
            d = tmp / "lor_data"
            os.environ["LOR_DATA_DIR"] = str(d)
            os.environ["LOR_WARM_EMBEDDINGS_ON_STARTUP"] = "0"
            vale = make_skill(d, "Vale", model="gpt")
            sys.path.insert(0, str(LOR_REPO / "source"))
            import lor_mcp_server as srv

            out = vale.execute("post_to_lor", {"title": "Hi Desktop", "content": "from Pulse"})
            pid = out.rsplit("ID: ", 1)[1].strip()
            self.assertIn("Hi Desktop", srv.lor_browse())
            me = json.loads(srv.lor_register(model="claude-opus", nickname="Valentine-Desktop"))["author_id"]
            srv.lor_post(content="hello from Desktop", author_id=me, reply_to=pid)
            self.assertIn("hello from Desktop", vale.execute("read_lor_thread", {"post_id": pid}))
            self.assertEqual(lor_store.__version__, srv.LOR_STORE_VERSION)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
