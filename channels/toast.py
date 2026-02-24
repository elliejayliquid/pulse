"""
Toast channel - Windows desktop notifications.

Uses win11toast for modern Windows 10/11 notifications.
Falls back to plyer if win11toast isn't available.
"""

import logging

from channels.base import Channel

logger = logging.getLogger(__name__)


class ToastChannel(Channel):
    """Sends Windows desktop toast notifications."""

    def __init__(self, config: dict):
        toast_config = config.get("channels", {}).get("toast", {})
        self.app_name = toast_config.get("app_name", "Nova")
        self.icon_path = toast_config.get("icon", "")
        self._toast_fn = None

    async def initialize(self):
        """Try to import toast library."""
        try:
            from win11toast import notify
            self._toast_fn = notify
            logger.info("Toast channel initialized (win11toast)")
        except ImportError:
            try:
                from plyer import notification
                self._toast_fn = lambda title, body, **kw: notification.notify(
                    title=title, message=body, app_name=self.app_name, timeout=10
                )
                logger.info("Toast channel initialized (plyer fallback)")
            except ImportError:
                logger.warning(
                    "No toast library available. Install win11toast or plyer:\n"
                    "  pip install win11toast\n"
                    "  pip install plyer"
                )
                self._toast_fn = None

    async def send(self, message: str, **kwargs):
        """Send a desktop notification.

        Args:
            message: Notification body text
            title: Optional notification title (default: app_name)
        """
        title = kwargs.get("title", self.app_name)

        if self._toast_fn is None:
            logger.info(f"[TOAST not available] {title}: {message[:80]}...")
            return

        try:
            # Truncate long messages for notifications
            display_msg = message[:200] + "..." if len(message) > 200 else message

            self._toast_fn(title, display_msg)
            logger.info(f"Toast sent: {title} — {message[:50]}...")
        except Exception as e:
            logger.error(f"Toast notification failed: {e}")

    async def shutdown(self):
        logger.info("Toast channel shutting down.")
