"""Pulse Engine desktop launcher/config manager."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from gui.api import PulseAPI


def main() -> int:
    parser = argparse.ArgumentParser(description="Pulse Engine desktop GUI")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the HTML shell in the default browser instead of pywebview.",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    index_path = root / "gui" / "index.html"

    if args.browser:
        webbrowser.open(index_path.as_uri())
        return 0

    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed. Install requirements or run:\n"
            "  python pulse_gui.py --browser\n\n"
            "The browser fallback is read-only/mock-backed unless pywebview is running.",
            file=sys.stderr,
        )
        webbrowser.open(index_path.as_uri())
        return 1

    api = PulseAPI(root)
    window = webview.create_window(
        "Pulse Engine",
        index_path.as_uri(),
        js_api=api,
        width=960,
        height=860,
        min_size=(760, 620),
    )
    api._window = window

    def on_closing():
        if api._force_close:
            return True
        running = api.get_running_personas()
        if running:
            api._close_requested = running
            return False
        return True

    window.events.closing += on_closing
    webview.start(debug=False, private_mode=True)
    return 0 if window else 1


if __name__ == "__main__":
    raise SystemExit(main())
