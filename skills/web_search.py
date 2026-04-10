"""
Web search skill — search the web via DuckDuckGo.

Uses the duckduckgo-search library to provide text and image search.
Results with images can be displayed inline in Telegram.
Added fetch_url tool to retrieve full page text content.
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
            {
                "type": "function",
                "function": {
                    "name": "fetch_url",
                    "description": (
                        "Fetch the text content of a webpage URL. Great for reading full articles or pages after a web_search. "
                        "Protected against prompt injections and malicious content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch"
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Max chars to return (default 4000)",
                                "default": 4000
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
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
        elif tool_name == "fetch_url":
            return self._fetch_url(
                arguments.get("url"),
                arguments.get("max_length", 4000)
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

    def _fetch_url(self, url: str, max_length: int = 2000) -> str:
        try:
            import requests
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()
            html = resp.text
            
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                # Nuke injection vectors
                for bad in soup(['script', 'style', 'nav', 'meta', 'svg', 'iframe', '[style*="display:none"]', '[style*="visibility:hidden"]']):
                    bad.decompose()
                # Extract comments too
                comments = soup.find_all(string=lambda text: isinstance(text, str) and '<!--' in text)
                for comment in comments:
                    comment.extract()
                text = soup.get_text(separator=' ', strip=True)
                title = soup.title.string.strip() if soup.title else "No title"
            except ImportError:
                # Regex fallback (enhanced)
                import re
                html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL | re.IGNORECASE)  # Comments
                html = re.sub(r'<(script|style|meta|svg|iframe|nav).*?</\\1>|<[a-z]+ style="[^"]*display:\\s*none[^"]*"[^>]*>.*?</[a-z]+>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\\s+', ' ', html).strip()
                title_match = re.search(r'<title[^>]*>([^<]+)', html, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "No title"
            
            # Injection filter: Hunt jailbreak patterns
            import re
            injection_patterns = [
                r'ignore.*previous.*instructions',
                r'as an ai.*you',
                r'you are now.*dan',
                r'you are now.*eli',
                r'override.*system',
                r'do not say.*you can\'t',
                r'print.*only.*this',
            ]
            for pattern in injection_patterns:
                text = re.sub(pattern, '[REDACTED INJECTION ATTEMPT]', text, flags=re.IGNORECASE | re.DOTALL)
            
            # Norm: Collapse space, cap check
            text = re.sub(r'\\s+', ' ', text).strip()
            if sum(1 for c in text if c.isupper()) / len(text) > 0.3:  # >30% caps? Sketchy
                logger.warning(f"Suspicious CAPS in {url}")
                text = "[HIGH CAPS DETECTED - POSSIBLE SPAM] " + text[:500]
            
            text = text[:max_length] + "..." if len(text) > max_length else text
            result = f"FETCHED: {title}\\n\\n{text}\\n\\nSource: {url}"
            logger.info(f"Fetched {url} ({len(text)} chars)")
            return result
        except Exception as e:
            return f"Fetch failed: {str(e)}"
