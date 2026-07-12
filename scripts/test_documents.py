"""Tests for the document inbox: core/documents.py + skills/documents.py.

    .venv/Scripts/python.exe scripts/test_documents.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.documents import extract_text, save_to_inbox  # noqa: E402
from skills.documents import SLICE_CHARS, DocumentsSkill  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}  {detail}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        inbox = data_dir / "inbox"
        config = {"paths": {"database": str(data_dir / "test.db")}}

        # save_to_inbox: sanitization + collision handling
        p1 = save_to_inbox(inbox, "notes.txt", b"hello world")
        p2 = save_to_inbox(inbox, "notes.txt", b"second file")
        p3 = save_to_inbox(inbox, "we?ird*na<me.md", b"# hi")
        check("saved with original name", p1.name == "notes.txt")
        check("collision renamed", p2.name == "notes_2.txt", p2.name)
        check("filename sanitized", "?" not in p3.name and "<" not in p3.name, p3.name)

        # extract_text: text, binary, unsupported
        text, note = extract_text(p1)
        check("text extraction", text == "hello world" and not note)
        binary = save_to_inbox(inbox, "blob.txt", b"\x00\x01\x02data")
        text, note = extract_text(binary)
        check("binary detected", not text and "binary" in note)
        exe = save_to_inbox(inbox, "tool.exe", b"MZ....")
        text, note = extract_text(exe)
        check("unsupported extension", not text and ".exe" in note, note)

        # PDF: blank page (no extractable text) exercises the pypdf path
        try:
            from pypdf import PdfWriter
            import io
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            buffer = io.BytesIO()
            writer.write(buffer)
            pdf = save_to_inbox(inbox, "scan.pdf", buffer.getvalue())
            text, note = extract_text(pdf)
            check("blank pdf gives honest note", not text and "no extractable text" in note, note)
        except ImportError:
            check("pypdf installed", False, "pypdf missing from venv")

        # Skill: list + read + slicing + traversal guard
        skill = DocumentsSkill(config)
        listing = skill.execute("list_documents", {})
        check("list shows files", "notes.txt" in listing and "notes_2.txt" in listing, listing)

        result = skill.execute("read_document", {"filename": "notes.txt"})
        check("read returns content", "hello world" in result and "End of document" in result)

        long_text = ("lorem ipsum " * 2000).strip()  # ~24k chars
        save_to_inbox(inbox, "long.txt", long_text.encode())
        first = skill.execute("read_document", {"filename": "long.txt"})
        check("long doc sliced", f"offset={SLICE_CHARS}" in first, first[-120:])
        second = skill.execute("read_document", {"filename": "long.txt", "offset": SLICE_CHARS})
        check("second slice continues", f"chars {SLICE_CHARS}-" in second, second[:80])

        result = skill.execute("read_document", {"filename": "../../secret.txt"})
        check("traversal blocked", "inbox" in result and "secret" not in result.replace("secret.txt", ""), result)
        result = skill.execute("read_document", {"filename": "nope.txt"})
        check("missing file handled", "No document named" in result)

        # Empty inbox message
        empty_skill = DocumentsSkill({"paths": {"database": str(Path(tmp) / "other" / "x.db")}})
        check("empty inbox message", "empty" in empty_skill.execute("list_documents", {}))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
