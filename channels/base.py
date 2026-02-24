"""
Base channel interface.

All output channels (LoR, toast, Telegram, etc.) implement this interface.
This makes it easy to add new channels without touching the engine.
"""

from abc import ABC, abstractmethod


class Channel(ABC):
    """Abstract base for output channels."""

    @abstractmethod
    async def send(self, message: str, **kwargs):
        """Send a message through this channel.

        Args:
            message: The message content
            **kwargs: Channel-specific options (e.g., category for LoR)
        """
        pass

    @abstractmethod
    async def initialize(self):
        """Set up the channel (called once at startup)."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Clean up the channel (called on exit)."""
        pass
