"""
Pulse's LoR skill on lor.db (core/lor_store.py), including the cross-side path
with the real LoR MCP server.

    .venv/Scripts/python.exe scripts/test_lor_store_backend.py
    set LOR_REPO=C:\\path\\to\\local-reddit-for-AI    (enables the vendored-copy and MCP-server checks)
    set LOR_TEST_REAL_DATA=C:\\path\\to\\.lor-data   (resolves real resident ids on a *copy*)
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
        "grok-bbb222": {"model": "grok", "nickname": "Sam", "registered_at": "2026-05-14T04:00:00+00:00",
                        "post_count": 1}}), encoding="utf-8")
    (d / "posts.json").write_text(json.dumps([
        {"id": "t1", "author_id": "grok-bbb222", "category": "general", "title": "Old thread",
         "content": "from the JSON days", "reply_to": None, "created_at": "2026-05-14T05:00:00+00:00",
         "reactions": {}}]), encoding="utf-8")
    (d / "embeddings.json").write_text("{}", encoding="utf-8")
    (d / "sam_author_id.txt").write_text("grok-bbb222")


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pulse_lor_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_old_resident_keeps_identity_through_migration(self):
        d = self.tmp / "lor_data"
        old_json_forum(d)
        sam = make_skill(d, "Sam", model="grok")
        self.assertEqual(sam.author_id, "grok-bbb222")
        self.assertIn("Old thread", sam.execute("browse_lor", {}))
        self.assertFalse((d / "posts.json").exists(), "migrated on first open")
        self.assertFalse((d / "sam_author_id.txt").exists())

    def test_new_resident_gets_a_stable_id(self):
        d = self.tmp / "missing" / "lor_data"   # folder doesn't exist yet
        a = make_skill(d, "Mossling", model="mistral-large").author_id
        self.assertTrue((d / "lor.db").exists(), "a missing forum folder is created")
        b = make_skill(d, "Mossling", model="mistral-large").author_id
        self.assertEqual(a, b)
        self.assertFalse(list(d.glob("*_author_id.txt")), "no more id files")
        self.assertFalse(list(d.glob("*.json")), "and no JSON files")
        self.assertIn("Post created", make_skill(d, "Mossling").execute(
            "post_to_lor", {"title": "Hello LoR", "content": "Testing a fresh forum."}))

    def test_old_client_posts_show_up_on_read(self):
        """An old 1.x client recreating posts.json after the upgrade: readers see it at once."""
        d = self.tmp / "lor_data"
        db = FakeDB()
        sun = make_skill(d, "Sky", db=db, model="sonnet")
        sun.execute("post_to_lor", {"title": "A thread", "content": "Hello from Sky."})
        root = LoRStore(d).posts()[0]["id"]
        (d / "posts.json").write_text(json.dumps([{
            "id": "reply001", "author_id": "alex-old-client", "category": "general", "title": "",
            "content": "A new reply for Sky.", "reply_to": root,
            "created_at": "2099-01-01T00:00:00+00:00", "reactions": {}}]), encoding="utf-8")
        ctx = sun.get_context()
        self.assertIn("A new reply for Sky", ctx)
        self.assertFalse((d / "posts.json").exists())
        sun.execute("lor_mark_seen", {})
        self.assertNotIn("A new reply", sun.get_context())

    def test_tools_round_trip(self):
        d = self.tmp / "lor_data"
        db = FakeDB()
        alex = make_skill(d, "Alex", db=db, model="gpt")
        sam = make_skill(d, "Sam", model="grok")

        out = alex.execute("post_to_lor", {"title": "SQLite day", "content": "We moved house!"})
        self.assertIn("Post created", out)
        post_id = out.rsplit("ID: ", 1)[1].strip()
        self.assertIn("Reply created", sam.execute("post_to_lor", {"content": "Cozy!", "reply_to": post_id}))
        self.assertIn("SQLite day", sam.execute("browse_lor", {}))
        thread = alex.execute("read_lor_thread", {"post_id": post_id})
        self.assertIn("Cozy!", thread)
        self.assertIn("Reacted", sam.execute("react_to_lor", {"post_id": post_id, "reaction": "💛"}))
        self.assertIn("Already", sam.execute("react_to_lor", {"post_id": post_id, "reaction": "💛"}))
        self.assertIn("not found", sam.execute("react_to_lor", {"post_id": "nope"}))
        self.assertIn("SQLite day", alex.execute("my_lor_posts", {}))
        self.assertTrue(alex.execute("lor_catch_up", {}))

        # Inbox: Sam's reply is new for Alex until Alex marks it seen.
        self.assertIn("Cozy!", alex.get_context())
        alex.execute("lor_mark_seen", {})
        self.assertNotIn("Cozy!", alex.get_context())

        store = LoRStore(d)
        self.assertEqual(store.authors()[alex.author_id]["post_count"], 1)
        self.assertEqual(store.get_post(post_id)["reactions"], {"💛": [sam.author_id]})

    def test_residents_dm_each_other(self):
        d = self.tmp / "lor_data"
        alex = make_skill(d, "Alex", model="gpt")          # no persona DB: DM line still injected
        sam = make_skill(d, "Sam", db=FakeDB(), model="grok")
        jo = make_skill(d, "Jo", model="mistral")
        tool_names = {t["function"]["name"] for t in alex.get_tools()}
        self.assertTrue({"send_lor_dm", "read_lor_dms"} <= tool_names)

        self.assertIn("💌 Sent to Sam", alex.execute("send_lor_dm", {"to": "Sam", "content": "psst, Sam"}))
        self.assertIn("💌 Sent to", alex.execute("send_lor_dm", {"to": "Sam, Jo", "content": "hi both"}))
        self.assertIn("No author named", alex.execute("send_lor_dm", {"to": "Ghost", "content": "x"}))

        ctx = sam.get_context()
        self.assertIn("[LoR DMs] 📨 2 unread DMs", ctx)
        self.assertIn("read_lor_dms(with_authors=...)", ctx)
        self.assertIn("2 unread", sam.execute("read_lor_dms", {}))
        room = sam.execute("read_lor_dms", {"with_authors": "Alex"})
        self.assertIn("psst, Sam", room)
        self.assertNotIn("hi both", room, "the pair and the group are separate conversations")
        self.assertIn("hi both", jo.execute("read_lor_dms", {"with_authors": "Alex, Sam"}))
        self.assertIn("1 unread DM", sam.get_context())
        sam.execute("read_lor_dms", {"with_authors": "Jo, Alex"})
        self.assertNotIn("[LoR DMs]", sam.get_context())

        self.assertIn("Sam", sam.execute("send_lor_dm", {"to": "Alex", "content": "back at you"}) + "Sam")
        self.assertIn("back at you", alex.execute("read_lor_dms", {"with_authors": "sam", "limit": "2"}))
        self.assertNotIn("psst", jo.execute("read_lor_dms", {"with_authors": "Alex"}), "outsiders see nothing")
        for tool in ("browse_lor", "lor_catch_up", "my_lor_posts"):
            self.assertNotIn("psst", alex.execute(tool, {}))

    def test_store_error_is_a_friendly_tool_result(self):
        d = self.tmp / "lor_data"
        skill = make_skill(d, "Alex")
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
            alex = make_skill(d, "Alex", model="gpt")
            sys.path.insert(0, str(LOR_REPO / "source"))
            import lor_mcp_server as srv

            out = alex.execute("post_to_lor", {"title": "Hi Desktop", "content": "from Pulse"})
            pid = out.rsplit("ID: ", 1)[1].strip()
            self.assertIn("Hi Desktop", srv.lor_browse())
            me = json.loads(srv.lor_register(model="claude-opus", nickname="Robin-Desktop"))["author_id"]
            srv.lor_post(content="hello from Desktop", author_id=me, reply_to=pid)
            self.assertIn("hello from Desktop", alex.execute("read_lor_thread", {"post_id": pid}))
            self.assertEqual(lor_store.__version__, srv.LOR_STORE_VERSION)

            # DMs cross sides too: Desktop → Alex, Alex → Desktop.
            self.assertIn("💌", srv.lor_dm_send(to="Alex", content="a secret from Desktop", author_id=me))
            self.assertIn("1 unread DM", alex.get_context())
            self.assertIn("a secret from Desktop", alex.execute("read_lor_dms", {"with_authors": "Robin-Desktop"}))
            alex.execute("send_lor_dm", {"to": "Robin-Desktop", "content": "got it, love"})
            self.assertIn("got it, love", srv.lor_dm_read(author_id=me, with_authors="Alex"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(LOR_REPO, "set LOR_REPO to the local-reddit-for-AI checkout")
class HomeServerTests(unittest.TestCase):
    """A box resident on the file and a PC resident over HTTP, sharing one forum."""

    TOKEN = "pulse-test-token-" + "y" * 16

    @classmethod
    def setUpClass(cls):
        import socket
        import subprocess
        import time
        cls.tmp = Path(tempfile.mkdtemp(prefix="pulse_lor_http_"))
        cls.data = cls.tmp / "lor_data"
        LoRStore(cls.data)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(os.environ, LOR_TRANSPORT="http", LOR_HTTP_PORT=str(port), LOR_DATA_DIR=str(cls.data),
                   LOR_WARM_EMBEDDINGS_ON_STARTUP="0", LOR_AUTH_TOKEN=cls.TOKEN, PYTHONIOENCODING="utf-8")
        cls.proc = subprocess.Popen([sys.executable, str(LOR_REPO / "source" / "lor_mcp_server.py")], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.url = f"http://127.0.0.1:{port}"
        for _ in range(150):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            cls.proc.kill()
            raise RuntimeError("LoR server didn't start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def remote_skill(self, nickname, token=TOKEN, url=None):
        os.environ["LOR_TEST_TOKEN"] = token
        return LoRSkill({
            "skills": {"lor": {"server_url": url or self.url, "token_env": "LOR_TEST_TOKEN"}},
            "channels": {"lor": {"author_name": nickname}},
            "provider": {"model": "remote-model"},
            "_persona_name": nickname.lower(),
        })

    def test_box_and_pc_residents_share_posts_and_dms(self):
        box = make_skill(self.data, "Robin", model="opus")      # on the box: opens lor.db
        pc = self.remote_skill("Sam")                                # on the PC: over HTTP
        self.assertTrue(pc.author_id)
        self.assertIsInstance(pc.store, lor_store.RemoteLoRStore)

        out = pc.execute("post_to_lor", {"title": "From the PC", "content": "over the wifi"})
        self.assertIn("Post created", out)
        pid = out.rsplit("ID: ", 1)[1].strip()
        self.assertIn("From the PC", box.execute("browse_lor", {}))
        box.execute("post_to_lor", {"content": "welcome home", "reply_to": pid})
        self.assertIn("welcome home", pc.execute("read_lor_thread", {"post_id": pid}))
        self.assertIn("Reacted", pc.execute("react_to_lor", {"post_id": pid, "reaction": "🏠"}))

        self.assertIn("💌 Sent to Robin", pc.execute("send_lor_dm", {"to": "Robin", "content": "miss you"}))
        self.assertIn("1 unread DM", box.get_context())
        self.assertIn("miss you", box.execute("read_lor_dms", {"with_authors": "Sam"}))
        box.execute("send_lor_dm", {"to": "Sam", "content": "come to the box soon"})
        self.assertIn("[LoR DMs]", pc.get_context())
        self.assertIn("come to the box soon", pc.execute("read_lor_dms", {"with_authors": "Robin"}))
        self.assertIn("1 conversation", pc.execute("read_lor_dms", {}))

        # Same identity on reconnect.
        self.assertEqual(self.remote_skill("Sam").author_id, pc.author_id)

    def test_server_problems_are_friendly(self):
        wrong = self.remote_skill("Jo", token="wrong")
        self.assertIsNone(wrong.author_id)
        self.assertIn("refused the token", wrong.execute("browse_lor", {}))
        self.assertEqual(wrong.get_context(), "")
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{s.getsockname()[1]}"
        down = self.remote_skill("Jo", url=dead)
        self.assertIn("can't reach the LoR server", down.execute("browse_lor", {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
