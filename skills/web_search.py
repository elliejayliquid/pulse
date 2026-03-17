"""
Web search skill — search the web via DuckDuckGo.

Uses the duckduckgo-search library to provide text and image search.
Results with images can be displayed inline in Telegram.
"""

import logging
from skills.base import BaseSkill

logger = logging.getLogger(__name__)


class WebSearchSkill(BaseSkill):
    name = "web_search"

    def __init__(self, config: dict):
        super().__init__(config)
        self.pending_images: list[str] = []  # URLs to send as photos after response
        self.pending_sources: list[dict] = []  # {title, url} for source attribution

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                        "Use this when your human asks you to look something up, find information, "
                        "check facts, or search for anything online."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Number of results to return (default 5, max 10)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "image_search",
                    "description": (
                        "Search for images using DuckDuckGo. Returns image URLs that will be "
                        "displayed inline in the chat. Use when your human asks to find or "
                        "show images of something."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search images for",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Number of images to return (default 3, max 5)",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "web_search":
            return self._web_search(
                query=arguments.get("query", ""),
                max_results=arguments.get("max_results", 5),
            )
        elif tool_name == "image_search":
            return self._image_search(
                query=arguments.get("query", ""),
                max_results=arguments.get("max_results", 3),
            )
        return f"Unknown tool: {tool_name}"

    def _web_search(self, query: str, max_results: int = 5) -> str:
        if not query.strip():
            return "Search query cannot be empty."

        max_results = max(1, min(max_results, 10))

        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except ImportError:
            return "Web search unavailable — ddgs package not installed."
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {e}"

        if not results:
            self.pending_sources.clear()
            return f"No results found for: {query}"

        # Track sources for attribution
        self.pending_sources.clear()
        lines = [
            f"WEB SEARCH RESULTS for: {query}",
            f"Found {len(results)} results. IMPORTANT INSTRUCTIONS:",
            "- Only mention information that appears in these results below.",
            "- Do NOT invent or hallucinate titles, authors, or facts not listed here.",
            "- If the user asked a question and the results don't answer it, say so.",
            "",
        ]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            if body:
                lines.append(f"   {body[:200]}")
            lines.append("")
            if url:
                self.pending_sources.append({"title": title, "url": url})

        logger.info(f"Web search: '{query}' -> {len(results)} results")
        return "\n".join(lines)

    def _image_search(self, query: str, max_results: int = 3) -> str:
        if not query.strip():
            return "Search query cannot be empty."

        max_results = max(1, min(max_results, 5))

        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results))
        except ImportError:
            return "Image search unavailable — ddgs package not installed."
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return f"Image search failed: {e}"

        if not results:
            return f"No images found for: {query}"

        # Queue images for Telegram display
        self.pending_images.clear()
        lines = [
            f"IMAGE SEARCH RESULTS for: {query}",
            f"Found {len(results)} images. These will be displayed to the user automatically.",
            "Just briefly describe what was found — do NOT list raw URLs.",
            "",
        ]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            image_url = r.get("image", "")
            source = r.get("source", "")
            lines.append(f"{i}. {title}")
            if source:
                lines.append(f"   Source: {source}")
            if image_url:
                self.pending_images.append(image_url)
            lines.append("")

        logger.info(f"Image search: '{query}' -> {len(results)} results, {len(self.pending_images)} images queued")
        return "\n".join(lines)
