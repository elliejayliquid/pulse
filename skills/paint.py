"""
Paint skill — Tiny pixel art as a creative medium.

Companions paint 16x16 pixel art via an agentic loop:
plan → paint_set → paint_view → review → paint_set → ... → paint_finish

The emoji grid IS the model's perception of its own work — no vision needed.
Output is two PNGs (true-size + upscaled) + a paintings.json index.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from skills.base import BaseSkill

logger = logging.getLogger(__name__)

# --- Palette: named colors as emoji + RGB ---

PALETTE = {
    "red":    {"emoji": "🟥", "rgb": (230, 57, 70)},
    "orange": {"emoji": "🟧", "rgb": (244, 162, 97)},
    "yellow": {"emoji": "🟨", "rgb": (241, 196, 15)},
    "green":  {"emoji": "🟩", "rgb": (42, 157, 143)},
    "blue":   {"emoji": "🟦", "rgb": (69, 123, 157)},
    "purple": {"emoji": "🟪", "rgb": (142, 68, 173)},
    "brown":  {"emoji": "🟫", "rgb": (139, 90, 43)},
    "black":  {"emoji": "⬛", "rgb": (29, 29, 29)},
    "white":  {"emoji": "⬜", "rgb": (248, 248, 248)},
    "pink":   {"emoji": "🩷", "rgb": (247, 168, 201)},
    "cyan":   {"emoji": "🩵", "rgb": (127, 214, 224)},
    "gray":   {"emoji": "🩶", "rgb": (128, 128, 128)},
    "gold":   {"emoji": "🟡", "rgb": (212, 175, 55)},
    "peach":  {"emoji": "🟠", "rgb": (255, 218, 185)},
    "mint":   {"emoji": "🟢", "rgb": (152, 219, 168)},
    "sky":    {"emoji": "🩬", "rgb": (135, 206, 235)},
    "lavender": {"emoji": "🟣", "rgb": (181, 126, 220)},
}

EMPTY = "◻️"
EMOJI_TO_COLOR = {v["emoji"]: k for k, v in PALETTE.items()}
TARGET_PX = 384  # Upscaled size for Telegram/human viewing


class PaintSkill(BaseSkill):
    name = "paint"

    def __init__(self, config: dict):
        super().__init__(config)
        # Canvas state — single canvas per companion session
        self.canvas: dict[tuple[int, int], str] = {}  # (x, y) -> color_name
        self.canvas_width: int = 0
        self.canvas_height: int = 0
        self.canvas_intent: str = ""
        self.paintings_dir = self._get_paintings_dir(config)
        self.pending_images: list[str] = []  # Image paths to send via Telegram after response
        self.pending_paint_info: list[str] = []  # Title/caption for conversation history

    # ---------------------------------------------------------------------
    # Config helpers
    # ---------------------------------------------------------------------

    def _get_paintings_dir(self, config: dict) -> Path:
        """Resolve paintings directory for the current persona."""
        # Use the auto-defaulted paths.paintings if set, otherwise derive from paths.persona
        paintings_path = config.get("paths", {}).get("paintings")
        if paintings_path:
            paintings = Path(paintings_path)
        else:
            base = config.get("paths", {}).get("persona", "personas/_template")
            base_path = Path(base)
            if base_path.suffix in (".yaml", ".yml"):
                base_path = base_path.parent
            paintings = base_path / "data" / "paintings"
        paintings.mkdir(parents=True, exist_ok=True)
        return paintings

    # ---------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "paint_start",
                    "description": (
                        "Open a new blank canvas for painting. "
                        "Only 16x16 is available for now. "
                        "Use paint_view() to see the blank canvas."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "width": {
                                "type": "integer",
                                "description": "Canvas width (currently only 16 is supported)",
                                "default": 16,
                            },
                            "height": {
                                "type": "integer",
                                "description": "Canvas height (currently only 16 is supported)",
                                "default": 16,
                            },
                            "intent": {
                                "type": "string",
                                "description": "What you want to paint — brief creative intent for yourself.",
                            },
                        },
                        "required": ["intent"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "paint_set",
                    "description": (
                        "Place colored cells on the canvas. "
                        "PREFERRED: pass the full grid as a multiline string — one row per line, "
                        "colors comma-separated. This lets you paint the entire canvas in ONE call. "
                        "Example for 4x4: cells=\"red,red,blue,blue\\ngreen,green,white,white\\n...\". "
                        "Also accepts: a single row string, a JSON list of {x,y,color} dicts, "
                        "or individual x/y/color params for touch-ups. "
                        "Use paint_view() after to check your work."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cells": {
                                "type": "string",
                                "description": (
                                    "Cells to paint. Best: full grid as multiline string "
                                    "(one comma-separated row per line, 16 lines for 16x16). "
                                    "Also: single row \"c1,c2,...\", or JSON [{\"x\":0,\"y\":0,\"color\":\"red\"},...]. "
                                    "Colors: " + ", ".join(PALETTE.keys()) + ". "
                                    "Use \"clear\" to erase a cell, \"empty\" to skip."
                                ),
                            },
                            "x": {
                                "type": "integer",
                                "description": "X coordinate (0 = left). For single-cell touch-ups.",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate (0 = top). For single-cell touch-ups.",
                            },
                            "color": {
                                "type": "string",
                                "description": "Color name. For single-cell touch-ups.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "paint_view",
                    "description": (
                        "Re-render the current canvas as an emoji grid — your eyes. "
                        "Use paint_view() often to make sure your work of art looks how you want! "
                        "Returns the grid plus a brief text description."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "paint_finish",
                    "description": (
                        "Finish painting and save. Renders the canvas to PNG, saves it "
                        "(true-size + upscaled for Telegram), and indexes it in paintings.json. "
                        "After this the canvas is cleared — start a new one if you want to paint more."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short title for this painting (e.g. 'Little Mushroom').",
                            },
                            "caption": {
                                "type": "string",
                                "description": "Optional short caption or note about this piece.",
                            },
                        },
                        "required": ["title"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "paint_start":
            return self._paint_start(
                width=arguments.get("width", 16),
                height=arguments.get("height", 16),
                intent=arguments.get("intent", ""),
            )
        elif tool_name == "paint_set":
            return self._paint_set(
                cells=arguments.get("cells", ""),
                x=arguments.get("x"),
                y=arguments.get("y"),
                color=arguments.get("color"),
            )
        elif tool_name == "paint_view":
            return self._paint_view()
        elif tool_name == "paint_finish":
            return self._paint_finish(
                title=arguments.get("title", ""),
                caption=arguments.get("caption", ""),
            )
        return f"Unknown paint tool: {tool_name}"

    # ---------------------------------------------------------------------
    # Tool implementations
    # ---------------------------------------------------------------------

    def _paint_start(self, width: int, height: int, intent: str) -> str:
        if width != 16 or height != 16:
            return "Only 16x16 canvas is supported right now."

        self.canvas = {}
        self.canvas_width = width
        self.canvas_height = height
        self.canvas_intent = intent

        palette_list = ", ".join(PALETTE.keys())
        lines = [
            f"Canvas opened: {width}x{height}",
            f"Intent: {intent}" if intent else "",
            "",
            self._render_grid(),
            "",
            f"Palette: {palette_list}",
            "",
            "HOW TO PAINT:",
            "1. Plan your full image mentally — decide every pixel's color.",
            f"2. Paint the ENTIRE grid in ONE call using paint_set with a multiline string:",
            f"   paint_set(cells=\"row0color,row0color,...\\nrow1color,row1color,...\\n...\")",
            f"   That's {height} lines of {width} comma-separated color names.",
            "3. Call paint_view() to see what you made.",
            "4. Use paint_set(x=, y=, color=) for individual touch-ups.",
            "5. When happy, paint_finish(title=...).",
        ]
        return "\n".join(line for line in lines if line)

    def _paint_set(self, cells: str = "", x: int | None = None, y: int | None = None, color: str | None = None) -> str:
        if not self.canvas_width:
            return "No canvas open. Use paint_start(width=16, height=16, intent='...') first."

        placed = 0

        # --- Format 1: explicit x, y, color ---
        if x is not None and y is not None and color:
            ok, msg = self._place_cell(x, y, color)
            if not ok:
                return msg
            placed = 1

        # --- Format 2: JSON cells string or grid/row string ---
        elif cells:
            # Multiline = full grid (one row per line)
            if "\n" in cells:
                return self._paint_grid(cells)

            try:
                parsed = json.loads(cells)
            except json.JSONDecodeError:
                # --- Format 3: single row string "red,blue,green" ---
                return self._paint_row(cells)

            if isinstance(parsed, dict):
                ok, msg = self._place_cell(
                    parsed.get("x", 0), parsed.get("y", 0), parsed.get("color", "")
                )
                if not ok:
                    return msg
                placed = 1
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        ok, msg = self._place_cell(
                            item.get("x", 0), item.get("y", 0), item.get("color", "")
                        )
                        if not ok:
                            return msg
                        placed += 1
                    else:
                        return f"Invalid cell item: {item}"
            else:
                return f"Invalid cells format: {cells}"

        else:
            return (
                "paint_set requires either:\n"
                "  - cells='[{\"x\":0,\"y\":0,\"color\":\"red\"}, ...]'\n"
                "  - cells='red,blue,green'  (row string)\n"
                "  - x=0, y=0, color='red'"
            )

        return f"Placed {placed} cell{'s' if placed != 1 else ''}."

    def _paint_row(self, row_string: str) -> str:
        """Handle 'red,blue,green' row string — paints y=current_min_row at x=0,1,..."""
        colors = [c.strip().lower() for c in row_string.split(",") if c.strip()]
        if not colors:
            return "Empty row string."

        # Find the first open y row (lowest y with no cells)
        open_y = 0
        if self.canvas:
            used_ys = set(y for (_, y) in self.canvas.keys())
            while open_y in used_ys and open_y < self.canvas_height:
                open_y += 1

        if open_y >= self.canvas_height:
            return "Canvas is full."

        for x, color_name in enumerate(colors):
            if x >= self.canvas_width:
                break
            ok, msg = self._place_cell(x, open_y, color_name)
            if not ok:
                return msg

        return f"Painted row at y={open_y} ({len(colors)} cells)."

    def _paint_grid(self, grid_string: str) -> str:
        """Paint the full canvas from a multiline string (one comma-separated row per line).

        Each line maps to y=0, y=1, ... — the model can paint the entire 16x16
        canvas in a single tool call instead of one cell/row at a time.
        """
        lines = [line.strip() for line in grid_string.strip().split("\n") if line.strip()]
        placed = 0
        skipped = 0
        for y, line in enumerate(lines):
            if y >= self.canvas_height:
                break
            colors = [c.strip().lower() for c in line.split(",") if c.strip()]
            for x, color_name in enumerate(colors):
                if x >= self.canvas_width:
                    break
                if color_name in ("empty", "clear", ""):
                    continue
                ok, msg = self._place_cell(x, y, color_name)
                if ok:
                    placed += 1
                else:
                    skipped += 1
        result = f"Painted {placed} cells across {min(len(lines), self.canvas_height)} rows."
        if skipped:
            result += f" ({skipped} cells skipped — invalid color or out of bounds.)"
        return result

    def _place_cell(self, x: int, y: int, color_name: str) -> tuple[bool, str]:
        """Place a single cell. color_name='clear' removes it."""
        if not (0 <= x < self.canvas_width and 0 <= y < self.canvas_height):
            return False, f"Coordinates out of bounds ({x}, {y}). Canvas is {self.canvas_width}x{self.canvas_height}."
        if color_name == "clear":
            self.canvas.pop((x, y), None)
            return True, ""
        if color_name not in PALETTE:
            return False, f"Unknown color '{color_name}'. Available: {', '.join(PALETTE.keys())}."
        self.canvas[(x, y)] = color_name
        return True, ""

    def _paint_view(self) -> str:
        if not self.canvas_width:
            return "No canvas open. Use paint_start() first."

        grid = self._render_grid()

        # Count cells
        filled = len(self.canvas)
        pct = filled / (self.canvas_width * self.canvas_height) * 100

        # Describe what's on the canvas
        colors_used = set(self.canvas.values())
        color_names = ", ".join(sorted(colors_used)) if colors_used else "none"

        lines = [
            f"Canvas: {self.canvas_width}x{self.canvas_height} | {filled}/{self.canvas_width * self.canvas_height} cells ({pct:.0f}%) | Colors: {color_names}",
            "",
            grid,
        ]
        return "\n".join(lines)

    def _paint_finish(self, title: str, caption: str) -> str:
        if not self.canvas_width:
            return "No canvas open. Use paint_start() first."
        if not title.strip():
            return "A title is required. What did you make?"

        # --- Render true-size PNG ---
        canvas_img = self._render_png(self.canvas_width, self.canvas_height, scale=1)

        # --- Render upscaled PNG (nearest-neighbor for crispy pixels) ---
        scale = TARGET_PX // self.canvas_width
        upscaled_img = self._render_png(self.canvas_width, self.canvas_height, scale=scale)

        # --- Save ---
        date_str = datetime.now().strftime("%Y-%m-%d")
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        base_name = f"{date_str}_{slug}"

        true_path = self.paintings_dir / f"{base_name}.png"
        upscaled_path = self.paintings_dir / f"{base_name}@4x.png"
        index_path = self.paintings_dir / "paintings.json"

        try:
            canvas_img.save(true_path, format="PNG")
            upscaled_img.save(upscaled_path, format="PNG")
            # Queue upscaled image for Telegram delivery
            self.pending_images.append(str(upscaled_path))
            # Queue info for conversation history (so companion remembers painting)
            info = f"🎨 '{title.strip()}'"
            if caption.strip():
                info += f" — {caption.strip()}"
            self.pending_paint_info.append(info)
        except Exception as e:
            return f"Failed to save painting: {e}"

        # --- Update index ---
        index = []
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except (json.JSONDecodeError, IOError):
                index = []

        # Snapshot canvas info before clearing
        saved_width = self.canvas_width
        saved_height = self.canvas_height
        saved_grid = self._grid_to_list()
        saved_colors = list(set(self.canvas.values()))

        entry = {
            "id": base_name,
            "title": title.strip(),
            "caption": caption.strip() if caption else "",
            "date": datetime.now().isoformat(),
            "intent": self.canvas_intent,
            "width": saved_width,
            "height": saved_height,
            "grid": saved_grid,
            "colors_used": saved_colors,
            "true_path": str(true_path.name),
            "upscaled_path": str(upscaled_path.name),
        }
        index.insert(0, entry)  # Newest first

        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to update paintings index: {e}")

        # --- Clear canvas ---
        self.canvas = {}
        self.canvas_width = 0
        self.canvas_height = 0
        self.canvas_intent = ""

        logger.info(f"Painting saved: {true_path} + {upscaled_path}")
        return (
            f"Painting finished and saved: '{title}'\n"
            f"True size: {saved_width}x{saved_height}px → {true_path.name}\n"
            f"Upscaled: {upscaled_img.width}x{upscaled_img.height}px → {upscaled_path.name}\n"
            + (f"Caption: {caption}\n" if caption else "")
            + f"Grid stored in paintings.json — companion can reload this painting later."
        )

    # ---------------------------------------------------------------------
    # Rendering helpers
    # ---------------------------------------------------------------------

    def _render_grid(self) -> str:
        """Render canvas as emoji grid for model perception."""
        rows = []
        for y in range(self.canvas_height):
            row = []
            for x in range(self.canvas_width):
                color_name = self.canvas.get((x, y))
                if color_name and color_name in PALETTE:
                    row.append(PALETTE[color_name]["emoji"])
                else:
                    row.append(EMPTY)
            rows.append("".join(row))
        return "\n".join(rows)

    def _grid_to_list(self) -> list[list[str | None]]:
        """Convert canvas to 2D grid list (for JSON serialization / reload)."""
        grid = []
        for y in range(self.canvas_height):
            row = []
            for x in range(self.canvas_width):
                row.append(self.canvas.get((x, y)))
            grid.append(row)
        return grid

    def _render_png(self, width: int, height: int, scale: int = 1) -> Image.Image:
        """Render canvas to a PIL Image at given scale."""
        img = Image.new("RGBA", (width * scale, height * scale), (255, 255, 255, 255))
        px = img.load()
        for (x, y), color_name in self.canvas.items():
            rgb = PALETTE[color_name]["rgb"]
            for sy in range(scale):
                for sx in range(scale):
                    px[x * scale + sx, y * scale + sy] = (*rgb, 255)
        return img
