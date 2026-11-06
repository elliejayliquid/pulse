"""
Memory Garden skill — plant memories as seedlings and watch them grow.
"""

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from skills.base import BaseSkill

logger = logging.getLogger(__name__)

# --- Constants ---

GRID_WIDTH = 12
GRID_HEIGHT = 8

# PNG snapshot layout (1x pixels; drawn at SNAP_SCALE and downsampled for smooth edges)
SNAP_SCALE = 2
SNAP_TILE = 72
SNAP_GAP = 6
SNAP_PAD = 32          # outer margin around the grid panel's tiles
SNAP_HEADER = 104      # top of the tile grid
SNAP_PANEL_INSET = 10  # grid panel padding around the tiles
SNAPSHOT_WIDTH = SNAP_PAD * 2 + GRID_WIDTH * SNAP_TILE + (GRID_WIDTH - 1) * SNAP_GAP
SNAPSHOT_HEIGHT = SNAP_HEADER + GRID_HEIGHT * SNAP_TILE + (GRID_HEIGHT - 1) * SNAP_GAP + SNAP_PAD

# Palette mirrors the desktop GUI tokens (gui/style.css :root)
GUI_BG = (13, 13, 15)
GUI_SURFACE = (22, 22, 26)
GUI_BORDER = (42, 42, 50)
GUI_TEXT_PRIMARY = (232, 230, 227)
GUI_TEXT_SECONDARY = (154, 152, 163)
GUI_TEXT_TERTIARY = (102, 101, 114)
GUI_ACCENT_DEFAULT = "#f97316"

# Fonts tried in order — Windows, Linux (the home server), macOS
BODY_FONTS = ["segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"]
BOLD_FONTS = ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"]
# Color emoji fonts. Noto is a bitmap font that only loads at size 109;
# we render big and scale down anyway. Linux: apt install fonts-noto-color-emoji
EMOJI_FONTS = ["seguiemj.ttf", "NotoColorEmoji.ttf", "Apple Color Emoji.ttc"]
EMOJI_RENDER_SIZE = 109

# Drawn instead of an emoji when no color emoji font exists — never a tofu box
FALLBACK_DOT_COLORS = {
    "🌱": (110, 190, 90), "🌿": (70, 160, 80), "✨": (250, 204, 21), "🥀": (140, 90, 70),
}

# Visual Stages
STAGE_SEEDLING = "🌱"
STAGE_SPROUT = "🌿"
STAGE_SHIMMER = "✨"
STAGE_WILTED = "🥀"
EMPTY_PLOT = "·"

# Bloom Pools by tag category
BLOOM_POOLS = {
    "personal": ["🌻", "🌷", "🌼", "💐"],
    "preference": ["🌻", "🌷", "🌼", "💐"],
    "relationship": ["🌹", "🌸", "🌺", "💐"],
    "emotion": ["🌹", "🌸", "🌺", "💐"],
    "creative": ["🌺", "🏵️", "🌸", "🌷"],
    "art": ["🌺", "🏵️", "🌸", "🌷"],
    "dream": ["🌺", "🏵️", "🌸", "🌷"],
    "project": ["🌳", "🌲", "🌴", "🌾"],
    "work": ["🌳", "🌲", "🌴", "🌾"],
    "knowledge": ["🌳", "🌲", "🌴", "🌾"],
    "chat_log": ["🍀", "☘️", "🌿", "🍃"],
    "session_log": ["🍀", "☘️", "🌿", "🍃"],
}

ALL_BLOOMS = [
    "🌻", "🌷", "🌼", "💐", "🌹", "🌸", "🌺", "🏵️", "🌳", "🌲", "🌴", "🌾", "🍀", "☘️", "🌿", "🍃"
]

class GardenSkill(BaseSkill):
    name = "garden"
    description = "Tend an emotional garden — plant, water, and grow memory flowers"
    search_summary = "Tend the emotional garden by planting, watering, pruning, and viewing memory flowers"
    search_examples = ["tend the garden", "water flowers", "plant a memory"]
    aliases = ["garden", "flowers", "plant", "water", "tend", "emotional garden", "housekeeping"]
    categories = ["creative", "reflection", "housekeeping"]

    def __init__(self, config: dict):
        super().__init__(config)
        self._db = config.get("_db")
        self._shared_db = config.get("_shared_db") or self._db
        self.garden_dir = self._get_garden_dir(config)
        self.pending_images: list[str] = []
        
        # Ensure garden directory exists
        self.garden_dir.mkdir(parents=True, exist_ok=True)

    def _get_garden_dir(self, config: dict) -> Path:
        """Resolve garden directory for snapshots."""
        garden_path = config.get("paths", {}).get("garden")
        if garden_path:
            return Path(garden_path)
        base = config.get("paths", {}).get("persona", "personas/_template")
        base_path = Path(base)
        if base_path.suffix in (".yaml", ".yml"):
            base_path = base_path.parent
        return base_path / "data" / "garden"

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "garden_plant",
                    "description": (
                        "Plant a memory as a seedling in your garden. "
                        "You must pick an empty plot (x, y) where x is 0-11 and y is 0-7. "
                        "Optionally give it a pet name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "integer",
                                "description": "The ID of the memory to plant (from save_memory or search_memory).",
                            },
                            "x": {"type": "integer", "description": "Horizontal position (0-11)."},
                            "y": {"type": "integer", "description": "Vertical position (0-7)."},
                            "name": {"type": "string", "description": "Optional name for this specific plant."},
                        },
                        "required": ["memory_id", "x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "garden_water",
                    "description": "Water a plant to boost its growth and restore its health. One plant per call.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Horizontal position."},
                            "y": {"type": "integer", "description": "Vertical position."},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "garden_prune",
                    "description": "Prune a plant to restore its health. Useful for wilted or neglected plants.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Horizontal position."},
                            "y": {"type": "integer", "description": "Vertical position."},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "garden_view",
                    "description": (
                        "View your garden as an emoji grid. "
                        "Set show=true to also send a pretty PNG snapshot to your human."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "show": {
                                "type": "boolean",
                                "description": "If true, queue a PNG snapshot for your human. Default: false.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "garden_info",
                    "description": "Get detailed stats about a specific plant at (x, y).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Horizontal position."},
                            "y": {"type": "integer", "description": "Vertical position."},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
        ]

    def get_context(self) -> str:
        """Inject garden status into prompt."""
        if not self._db:
            return ""
        
        # Run a tick to sync growth/health
        self._tick_all()
        
        plants = self._db.get_all_plants()
        if not plants:
            return "🌱 Garden: Your garden is empty. Why not plant a memory?"
        
        wilted = [p for p in plants if p['health'] < 0.3]
        needs_water = [p for p in plants if self._can_water(p)]
        
        status = f"🌱 Garden: {len(plants)} plants."
        if wilted:
            status += f" {len(wilted)} are wilting 🥀."
        if needs_water:
            status += f" {len(needs_water)} need water."
        
        return status

    def execute(self, tool_name: str, arguments: dict) -> str:
        if not self._db:
            return "Garden database not available."
        
        # Always tick before any operation
        self._tick_all()

        if tool_name == "garden_plant":
            return self._garden_plant(
                memory_id=arguments.get("memory_id"),
                x=arguments.get("x"),
                y=arguments.get("y"),
                name=arguments.get("name")
            )
        elif tool_name == "garden_water":
            return self._garden_water(arguments.get("x"), arguments.get("y"))
        elif tool_name == "garden_prune":
            return self._garden_prune(arguments.get("x"), arguments.get("y"))
        elif tool_name == "garden_view":
            return self._garden_view(show=arguments.get("show", False))
        elif tool_name == "garden_info":
            return self._garden_info(arguments.get("x"), arguments.get("y"))
        
        return f"Unknown garden tool: {tool_name}"

    # --- Tool Implementations ---

    def _garden_plant(self, memory_id: int, x: int, y: int, name: Optional[str]) -> str:
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return f"Position ({x}, {y}) is out of bounds (12x8 grid)."
        
        if self._db.get_plant(x, y):
            return f"There is already a plant at ({x}, {y}). Choose an empty plot."

        memory = self._shared_db.get_memory(memory_id)
        if not memory:
            return f"Memory ID {memory_id} not found."

        # Dedup — don't plant the same memory twice
        existing = [p for p in self._db.get_all_plants() if p['memory_id'] == memory_id]
        if existing:
            p = existing[0]
            return (
                f"Memory {memory_id} is already planted at ({p['x']}, {p['y']}) "
                f"as '{p.get('name') or 'unnamed'}'. Each memory can only be planted once."
            )

        # Name uniqueness — no two plants should share a name
        if name:
            same_name = [p for p in self._db.get_all_plants()
                         if p.get('name') and p['name'].lower() == name.lower()]
            if same_name:
                p = same_name[0]
                return (
                    f"A plant named '{p['name']}' already exists at ({p['x']}, {p['y']}). "
                    f"Pick a unique name for this one!"
                )

        # Determine species based on tags
        tags = memory.get("tags", [])
        species = "wildflower"
        for cat in BLOOM_POOLS:
            if cat in tags:
                species = cat
                break
        
        try:
            self._db.save_plant(x, y, memory_id, species, name)
            return f"Planted a new seedling at ({x}, {y}) linked to memory: '{memory['text'][:50]}...'"
        except Exception as e:
            return f"Failed to plant: {e}"

    def _garden_water(self, x: int, y: int) -> str:
        plant = self._db.get_plant(x, y)
        if not plant:
            return f"No plant at ({x}, {y})."
        
        if not self._can_water(plant):
            return f"You've already watered '{plant.get('name') or 'the plant'}' recently. Wait a bit!"

        new_growth = min(4.0, plant['growth'] + 0.3)
        self._db.update_plant(x, y, growth=new_growth, health=1.0, last_watered="datetime('now')")
        
        msg = f"You watered '{plant.get('name') or 'the plant'}' at ({x}, {y})."
        if new_growth >= 2.0 and plant['growth'] < 2.0:
            msg += "\nWait... something is happening! It's starting to bloom! ✨"
        
        return msg

    def _garden_prune(self, x: int, y: int) -> str:
        plant = self._db.get_plant(x, y)
        if not plant:
            return f"No plant at ({x}, {y})."
        
        self._db.update_plant(x, y, health=1.0)
        return f"You pruned '{plant.get('name') or 'the plant'}' at ({x}, {y}). It looks much healthier now!"

    def _garden_view(self, show: bool = False) -> str:
        plants = { (p['x'], p['y']): p for p in self._db.get_all_plants() }
        
        grid_lines = []
        for y in range(GRID_HEIGHT):
            row = []
            for x in range(GRID_WIDTH):
                plant = plants.get((x, y))
                if plant:
                    row.append(self._get_plant_emoji(plant))
                else:
                    row.append(EMPTY_PLOT)
            grid_lines.append(" ".join(row))
        
        grid_text = "\n".join(grid_lines)
        
        if show:
            # Render PNG snapshot and queue it for the human
            snapshot_path = self._render_snapshot(list(plants.values()))
            if snapshot_path:
                self.pending_images.append(str(snapshot_path))
            return f"Your Garden ({len(plants)} plants):\n\n{grid_text}\n\n(A PNG snapshot has been queued for your human.)"
        
        return f"Your Garden ({len(plants)} plants):\n\n{grid_text}"

    def _garden_info(self, x: int, y: int) -> str:
        plant = self._db.get_plant(x, y)
        if not plant:
            return f"No plant at ({x}, {y})."
        
        memory = self._shared_db.get_memory(plant['memory_id'])
        memory_text = memory['text'] if memory else "Unknown memory"
        
        age_delta = datetime.now(timezone.utc) - _parse_utc(plant['planted_at'])
        
        lines = [
            f"--- Plant Info ({x}, {y}) ---",
            f"Name: {plant.get('name') or 'Unnamed'}",
            f"Stage: {self._get_stage_name(plant)}",
            f"Growth: {plant['growth']:.2f} / 4.0",
            f"Health: {plant['health']:.2f} / 1.0",
            f"Planted: {age_delta.days} days ago",
            f"Memory: \"{memory_text[:100]}...\"",
            f"Last Tended: {plant['last_tended']}",
        ]
        return "\n".join(lines)

    # --- Internal Helpers ---

    def _tick_all(self):
        """Update growth and health for all plants based on time elapsed."""
        plants = self._db.get_all_plants()
        if not plants:
            return
            
        now_utc = datetime.now(timezone.utc)
        plant_map = {(p['x'], p['y']): p for p in plants}
        
        for p in plants:
            last_tended_str = p['last_tended']
            # Normalize and parse. We expect UTC from both SQLite and our Python calls.
            last_tended = datetime.fromisoformat(last_tended_str.replace(" ", "T"))
            if last_tended.tzinfo is None:
                last_tended = last_tended.replace(tzinfo=timezone.utc)
            
            delta = now_utc - last_tended
            days = delta.total_seconds() / 86400
            
            growth_inc = 0.0
            boost_applied = False
            
            # 1. Retrieval Boost (Cross-DB logic)
            # Check if linked memory was accessed since this plant was last updated.
            # We do this FIRST because we want boost even if passive growth time is small.
            if p.get('memory_id') and self._shared_db:
                memory = self._shared_db.get_memory(p['memory_id'])
                if memory and memory.get('last_accessed'):
                    # Normalize SQLite 'now' (YYYY-MM-DD HH:MM:SS) to ISO
                    acc_str = memory['last_accessed'].replace(" ", "T")
                    last_accessed = datetime.fromisoformat(acc_str)
                    if last_accessed.tzinfo is None:
                        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                    
                    if last_accessed > last_tended:
                        growth_inc += 0.15
                        boost_applied = True
            
            # Use a tiny threshold to avoid spamming updates if heartbeats are fast
            # but ONLY if no boost was applied.
            if days <= 0.0001 and not boost_applied: 
                continue
            
            # 2. Natural Growth (+0.1 per day)
            growth_inc += 0.1 * days
            
            # 3. Adjacent Bonus (+0.02 per day per matching neighbor)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = plant_map.get((p['x'] + dx, p['y'] + dy))
                if neighbor and neighbor['species'] == p['species']:
                    growth_inc += 0.02 * days
            
            new_growth = p['growth'] + growth_inc
            
            # 4. Health Decay
            new_health = p['health']
            if delta > timedelta(days=7):
                # -0.05 per day after 7 days of neglect
                decay_days = (delta - timedelta(days=7)).total_seconds() / 86400
                new_health = max(0.1, p['health'] - (0.05 * decay_days))
            
            # 5. Handle Surprise Bloom Assignment
            bloom_emoji = p['bloom_emoji']
            if new_growth >= 2.0 and not bloom_emoji:
                pool = BLOOM_POOLS.get(p['species'], ALL_BLOOMS)
                bloom_emoji = random.choice(pool)
            
            # Clamp growth
            new_growth = min(4.0, new_growth)
            
            # Update DB
            # We use UTC isoformat to match SQLite's datetime('now') and ensure
            # strict ordering with microsecond precision.
            self._db.update_plant(p['x'], p['y'], 
                                  growth=new_growth, 
                                  health=new_health, 
                                  bloom_emoji=bloom_emoji,
                                  last_tended=datetime.now(timezone.utc).isoformat())

    def _can_water(self, plant: dict) -> bool:
        if not plant.get('last_watered'):
            return True
        # DB timestamps are UTC — compare in UTC, not local time
        last = _parse_utc(plant['last_watered'])
        return (datetime.now(timezone.utc) - last) > timedelta(hours=8)

    def _get_plant_emoji(self, plant: dict) -> str:
        if plant['health'] < 0.3:
            return STAGE_WILTED
        
        g = plant['growth']
        if g < 1.0:
            return STAGE_SEEDLING
        elif g < 1.7:
            return STAGE_SPROUT
        elif g < 2.0:
            return STAGE_SHIMMER
        else:
            return plant['bloom_emoji'] or STAGE_SHIMMER

    def _get_stage_name(self, plant: dict) -> str:
        if plant['health'] < 0.3:
            return "Wilted"
        g = plant['growth']
        if g < 1.0: return "Seedling"
        if g < 1.7: return "Sprout"
        if g < 2.0: return "Shimmering"
        return "Bloomed"

    # --- Snapshot rendering ---

    def _render_snapshot(self, plants: list[dict]) -> Optional[Path]:
        """Render the garden as a PNG styled like the desktop GUI's garden card."""
        try:
            img = self._draw_snapshot(plants)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.garden_dir / f"garden_{timestamp}.png"
            img.save(path)

            # Ring buffer: keep only last 5 snapshots to avoid bloat
            snapshots = sorted(self.garden_dir.glob("garden_*.png"), key=lambda x: x.stat().st_mtime)
            if len(snapshots) > 5:
                for old in snapshots[:-5]:
                    old.unlink()

            return path
        except Exception as e:
            logger.error(f"Failed to render garden snapshot: {e}")
            return None

    def _draw_snapshot(self, plants: list[dict]) -> Image.Image:
        S = SNAP_SCALE

        def px(v: float) -> int:
            return round(v * S)

        accent = _parse_hex(self.config.get("accent"), GUI_ACCENT_DEFAULT)
        img = Image.new("RGB", (px(SNAPSHOT_WIDTH), px(SNAPSHOT_HEIGHT)), GUI_SURFACE)
        draw = ImageDraw.Draw(img)

        font_title = _load_font(BOLD_FONTS, px(24))
        font_stats = _load_font(BODY_FONTS, px(13))
        font_name = _load_font(BODY_FONTS, px(10))

        # Header: title left, stats right on a shared baseline (like .skill-garden-head)
        persona = str(self.config.get("_persona_name") or "Companion")
        title = f"{persona[:1].upper()}{persona[1:]}'s Memory Garden"
        draw.text((px(SNAP_PAD), px(56)), title, fill=GUI_TEXT_PRIMARY, font=font_title, anchor="ls")
        draw.text((px(SNAPSHOT_WIDTH - SNAP_PAD), px(56)), self._snapshot_stats(plants),
                  fill=GUI_TEXT_SECONDARY, font=font_stats, anchor="rs")
        draw.rounded_rectangle([px(SNAP_PAD), px(66), px(SNAP_PAD + 36), px(69)], radius=px(1.5), fill=accent)

        # Grid panel
        draw.rounded_rectangle(
            [px(SNAP_PAD - SNAP_PANEL_INSET), px(SNAP_HEADER - SNAP_PANEL_INSET),
             px(SNAPSHOT_WIDTH - SNAP_PAD + SNAP_PANEL_INSET), px(SNAPSHOT_HEIGHT - SNAP_PAD + SNAP_PANEL_INSET)],
            radius=px(12), fill=GUI_BG, outline=GUI_BORDER, width=px(1),
        )

        white = (255, 255, 255)
        plant_map = {(p["x"], p["y"]): p for p in plants}
        for gy in range(GRID_HEIGHT):
            for gx in range(GRID_WIDTH):
                x0 = SNAP_PAD + gx * (SNAP_TILE + SNAP_GAP)
                y0 = SNAP_HEADER + gy * (SNAP_TILE + SNAP_GAP)
                box = [px(x0), px(y0), px(x0 + SNAP_TILE) - 1, px(y0 + SNAP_TILE) - 1]
                cx = x0 + SNAP_TILE / 2
                p = plant_map.get((gx, gy))

                if not p:
                    draw.rounded_rectangle(box, radius=px(6), fill=_mix(white, GUI_BG, 0.012),
                                           outline=_mix(white, GUI_BG, 0.05), width=px(1))
                    cy = y0 + SNAP_TILE / 2
                    r = px(1.6)
                    draw.ellipse([px(cx) - r, px(cy) - r, px(cx) + r, px(cy) + r],
                                 fill=_mix(white, GUI_BG, 0.2))
                    continue

                stage = self._get_stage_name(p)
                if stage == "Bloomed":
                    fill, outline = _mix(accent, GUI_BG, 0.12), _mix(accent, GUI_BG, 0.45)
                elif stage == "Wilted":
                    fill, outline = _mix(white, GUI_BG, 0.02), GUI_BORDER
                else:
                    fill, outline = _mix(accent, GUI_BG, 0.06), _mix(accent, GUI_BG, 0.18)
                draw.rounded_rectangle(box, radius=px(6), fill=fill, outline=outline, width=px(1))

                name = (p.get("name") or "").strip()
                emoji = self._get_plant_emoji(p)
                size = px(SNAP_TILE * (0.48 if name else 0.56))
                cy = y0 + SNAP_TILE * (0.42 if name else 0.5)
                glyph = _emoji_image(emoji, size)
                if glyph is not None:
                    if stage == "Wilted":
                        glyph = glyph.copy()
                        glyph.putalpha(glyph.getchannel("A").point(lambda a: int(a * 0.65)))
                    img.paste(glyph, (px(cx) - glyph.width // 2, px(cy) - glyph.height // 2), glyph)
                else:
                    dot = FALLBACK_DOT_COLORS.get(emoji, accent)
                    r = size * 0.3
                    draw.ellipse([px(cx) - r, px(cy) - r, px(cx) + r, px(cy) + r], fill=dot)

                if name:
                    label = _fit_text(draw, name, font_name, px(SNAP_TILE - 8))
                    draw.text((px(cx), px(y0 + SNAP_TILE * 0.88)), label,
                              fill=GUI_TEXT_SECONDARY, font=font_name, anchor="ms")

        return img.resize((SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT), Image.LANCZOS)

    def _snapshot_stats(self, plants: list[dict]) -> str:
        if not plants:
            return "Nothing planted yet"
        n = len(plants)
        bloomed = sum(1 for p in plants if self._get_stage_name(p) == "Bloomed")
        thirsty = sum(1 for p in plants if self._can_water(p))
        wilting = sum(1 for p in plants if p["health"] < 0.3)
        parts = [f"{n} plant{'s' if n != 1 else ''}"]
        if bloomed:
            parts.append(f"{bloomed} bloomed")
        if thirsty:
            parts.append(f"{thirsty} can water")
        if wilting:
            parts.append(f"{wilting} wilting")
        return "  ·  ".join(parts)


# --- Rendering helpers (module-level caches: fonts and glyphs never change) ---

_FONT_CACHE: dict = {}
_EMOJI_CACHE: dict = {}


def _parse_utc(value) -> datetime:
    """Parse a DB timestamp (SQLite 'YYYY-MM-DD HH:MM:SS' or ISO) as UTC."""
    dt = datetime.fromisoformat(str(value).replace(" ", "T"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_hex(value, default: str) -> tuple:
    for candidate in (value, default):
        s = str(candidate or "").strip().lstrip("#")
        if len(s) == 6:
            try:
                return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
    return (249, 115, 22)


def _mix(color: tuple, base: tuple, alpha: float) -> tuple:
    """Flatten rgba(color, alpha) over an opaque base."""
    return tuple(round(c * alpha + b * (1 - alpha)) for c, b in zip(color, base))


def _load_font(candidates: list[str], size: int):
    key = (tuple(candidates), size)
    if key not in _FONT_CACHE:
        font = None
        for name in candidates:
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1
                font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def _emoji_font():
    if "emoji" not in _FONT_CACHE:
        font = None
        for name in EMOJI_FONTS:
            try:
                font = ImageFont.truetype(name, EMOJI_RENDER_SIZE)
                break
            except OSError:
                continue
        if font is None:
            logger.warning("Garden snapshot: no color emoji font found "
                           "(Linux: apt install fonts-noto-color-emoji) - drawing dots instead")
        _FONT_CACHE["emoji"] = font
    return _FONT_CACHE["emoji"]


def _emoji_image(emoji: str, size: int) -> Optional[Image.Image]:
    """Render one color emoji, trimmed to its visible pixels and fit into size x size.

    Centering on the trimmed bitmap instead of font metrics is what keeps
    every flower in the middle of its tile, whichever emoji font is used.
    """
    key = (emoji, size)
    if key in _EMOJI_CACHE:
        return _EMOJI_CACHE[key]
    result = None
    font = _emoji_font()
    if font is not None:
        glyph = emoji.replace("️", "")  # the variation selector draws as tofu without raqm
        canvas = Image.new("RGBA", (EMOJI_RENDER_SIZE * 3, EMOJI_RENDER_SIZE * 3), (0, 0, 0, 0))
        try:
            ImageDraw.Draw(canvas).text((EMOJI_RENDER_SIZE // 2, EMOJI_RENDER_SIZE // 2), glyph,
                                        font=font, embedded_color=True)
            bbox = canvas.getchannel("A").getbbox()
        except Exception as e:
            logger.debug(f"Garden snapshot: could not draw {emoji!r}: {e}")
            bbox = None
        if bbox:
            trimmed = canvas.crop(bbox)
            scale = size / max(trimmed.size)
            result = trimmed.resize((max(1, round(trimmed.width * scale)),
                                     max(1, round(trimmed.height * scale))), Image.LANCZOS)
    _EMOJI_CACHE[key] = result
    return result


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text.rstrip() + "…"
