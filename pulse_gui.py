"""Pulse Engine desktop launcher/config manager."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import os
import sys
import webbrowser
from pathlib import Path

from gui.api import PulseAPI


ERROR_ALREADY_EXISTS = 183


class SingleInstanceGuard:
    """Keep only one Pulse GUI process alive at a time."""

    def __init__(self, name: str):
        self.name = name
        self._handle = None
        self._lock_path = None

    def acquire(self) -> bool:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = [
                ctypes.wintypes.LPVOID,
                ctypes.wintypes.BOOL,
                ctypes.wintypes.LPCWSTR,
            ]
            kernel32.CreateMutexW.restype = ctypes.wintypes.HANDLE

            self._handle = kernel32.CreateMutexW(None, True, self.name)
            if not self._handle:
                raise ctypes.WinError(ctypes.get_last_error())
            return ctypes.get_last_error() != ERROR_ALREADY_EXISTS

        lock_path = Path(os.getenv("TMPDIR") or "/tmp") / f"{self.name}.lock"
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        self._handle = fd
        self._lock_path = lock_path
        os.write(fd, str(os.getpid()).encode("ascii"))
        return True

    def release(self) -> None:
        if not self._handle:
            return
        if sys.platform == "win32":
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._handle)
        else:
            os.close(self._handle)
            if self._lock_path:
                try:
                    self._lock_path.unlink()
                except OSError:
                    pass
        self._handle = None
        self._lock_path = None


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

    guard = SingleInstanceGuard("PulseEngineGui")
    if not guard.acquire():
        print("Pulse Engine GUI is already running.")
        return 0

    if args.browser:
        webbrowser.open(index_path.as_uri())
        guard.release()
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
        guard.release()
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
    try:
        webview.start(debug=False, private_mode=True)
        return 0 if window else 1
    finally:
        guard.release()


if __name__ == "__main__":
    raise SystemExit(main())
