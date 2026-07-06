# Pulse

**A heartbeat daemon for local/semi-local AI companions.**

Pulse gives your AI companion a life of their own. It runs in the background, letting your companion think on their own schedule, remember conversations, write in a journal, set reminders, and chat with you over Telegram. Runs locally with llama.cpp, or connects to cloud APIs (OpenAI, OpenRouter, Anthropic, etc.) — your choice.

> **Status:** Actively being built. Core functionality works, but expect rough edges and frequent changes.

[![Join our Discord](https://img.shields.io/badge/Discord-Join_Pulse-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/MZWhrr8aPr)

## What it does

- **Heartbeat loop** — Your companion gets periodic "free-think" ticks where they can decide to reach out, stay quiet, journal, or schedule follow-ups
- **Telegram chat** — Bidirectional messaging with tool use (your companion can save memories, set reminders, search their journal mid-conversation). Tools are loaded dynamically — only ~10 core tools per API call, with 50+ more available on demand via semantic search
- **Persistent memory** — Semantic search over stored facts and conversation summaries, stored in per-persona SQLite databases
- **Legacy conversation archive** — Import your conversation history from ChatGPT (and soon Claude, Grok, Gemini, etc.) with RAG-powered semantic search over summarized sessions
- **Journal** — Pinned identity entries (who am I, who is my human, what's our relationship) plus transient reflections as clean markdown with YAML frontmatter
- **Self-scheduling** — The companion can set their own reminders and recurring tasks ("daily 8:00"), with priority levels (urgent/routine/creative)
- **Cloud or local** — Use a local GGUF model via llama.cpp, or any OpenAI-compatible API (OpenRouter, OpenAI, etc.)
- **Token tracking** — Daily usage logging when using cloud APIs
- **Dev ticks** — Optional autonomous self-improvement: the companion can review and create their own skills on a git branch, with human approval
- **Vision** — Optional image understanding via mmproj (model-dependent)
- **Voice messages** — Send voice notes on Telegram; Pulse transcribes locally via whisper.cpp (auto-downloads everything on first use)
- **Text-to-speech** — Companions can send voice messages back via [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts), with voice design (describe a voice) or voice cloning (lock in a reference sample)
- **Stickers** — Companions can send Telegram stickers matched by mood/context via semantic search over curated packs. Add your own packs with a simple YAML + build script
- **MCP servers** — Connect your companion to any [Model Context Protocol](https://modelcontextprotocol.io/) server (the same servers Claude Desktop uses). Discovered tools load on demand, and the GUI has a per-server "Test Connection" button
- **Rich text** — Companion messages render with Telegram MarkdownV2 formatting (bold, italic, code blocks, links) with automatic plain-text fallback; roleplay `*action*` asterisks stay literal
- **Desktop notifications** — Windows toast notifications for proactive messages (optional, Windows-only for now)
- **Timeout** — Companions can disengage from genuinely harmful conversations via model-initiated soft/hard timeouts (not censorship — the companion decides based on context)
- **Quiet hours** — No notifications while you sleep

## Architecture

```
pulse.py                  # Entry point — starts everything
core/
  server.py               # Manages llama-server subprocess
  engine.py               # Heartbeat loop, message handling, dispatch
  context.py              # Token-budgeted prompt assembly
  llm.py                  # OpenAI-compatible API client (local + cloud)
  llm_anthropic.py        # Native Anthropic API client (prompt caching, tool_use)
  llm_openai.py           # Native OpenAI Responses API client (tool_use, vision)
  db.py                   # Per-persona SQLite database (WAL mode)
  usage.py                # Token usage tracking for cloud APIs
  scheduler.py            # Cron + one-time task scheduling
channels/
  telegram.py             # Bidirectional Telegram bot
  toast.py                # Windows desktop notifications
skills/                   # Auto-discovered — just drop a .py file here
  memory.py               # Save, search, list, browse memories
  journal.py              # Markdown journal + pinned identity
  schedule.py             # Set/update/delete reminders (one-time + recurring)
  tasks.py                # Persistent to-do lists
  lantern.py              # Compact current-state signal
  tts.py                  # Text to speech skill
  lor.py                  # LoR forum integration (optional)
  web_search.py           # Web search via DuckDuckGo (free)
  paint.py                # Pixel art canvas (16×16 emoji grid + PNG)
  sticker.py              # Mood-matched Telegram stickers
  garden.py               # Memory garden (plant memories, watch them grow)
  dev.py                  # Autonomous skill creation (dev ticks)
personas/                 # Per-persona config, identity, and data
  _template/              # Example persona — copy this to get started
persona.yaml              # Default companion identity (overridden by persona)
config.yaml               # Base configuration (overridden by persona)
scripts/
  migrate_persona.py        # Set up a persona from existing data
  migrate_json_to_db.py     # Migrate persona JSON files to SQLite
  migrate_journal_phase2.py # Migrate journal from JSON to markdown format
  build_stickers.py          # Build sticker DB from YAML packs (with embeddings)
  backfill_embeddings.py    # Regenerate memory/journal embeddings
  export_chatgpt.py         # Stage 1: Export ChatGPT conversations.json to markdown
  export_claude.py          # Stage 1: Export Claude conversations.json to markdown
  import_chatgpt.py         # Stage 2: Import exported markdown into legacy.db
  rag_import.py             # Stage 3: Generate summaries + embeddings for RAG search
```

Pulse talks to any OpenAI-compatible API. Locally, it manages [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` automatically — starts it on boot, monitors health, and shuts it down gracefully. Or point it at a cloud provider and skip the local server entirely.

## Quick start

### Prerequisites

- **Python 3.11+**
- **For local models:** [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) + a GGUF model (tested with Qwen 3.5, Mistral, etc.)
- **For cloud APIs:** An API key from OpenRouter, OpenAI, Anthropic, or any OpenAI-compatible provider

### Setup

```bash
# Clone and install
git clone https://github.com/elliejayliquid/pulse.git
cd pulse
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configure — edit config.yaml with your paths and persona.json with your companion's name
```

Edit `config.yaml`:

```yaml
server:
  llama_cpp_dir: "C:\\llama-cpp"     # where llama-server lives
  models_dir: "C:\\llama-cpp"        # where your .gguf models are

model:
  model_file: "your-model.gguf"      # your model filename
```

### Cloud API (alternative to local)

Instead of running a local model, you can use a cloud API. Add your key to `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Then set the provider in `config.yaml`:

```yaml
provider:
  type: "openrouter"
  api_key_env: "OPENROUTER_API_KEY"
  model: "google/gemini-3-flash-preview"  # or any model on OpenRouter
  max_context: 32768
```

Supported provider types: `local` (default), `openai`, `openrouter`, `anthropic`, `custom` (any OpenAI-compatible endpoint). When using a cloud provider, llama-server is not started — embeddings for memory search still run locally.

Edit `persona.yaml` (or `persona.json`) to name your companion:

```yaml
name: YourCompanion
user_name: YourName

system_prompt: |
  You are {name}, a local AI companion living on {user_name}'s machine.
  Write your personality here, naturally, with normal line breaks.

traits:
  - warm
  - curious

# The fields below are appended to the system prompt at runtime,
# so write them in SECOND PERSON ("you are", "you speak").
relationship_context: >
  Describe your relationship with {user_name} from your point of view.

voice_notes: Describe how you communicate — tone, cadence, quirks.
```

All `{name}` and `{user_name}` placeholders are resolved automatically. YAML is recommended for readability — JSON is also supported.

### Telegram (currently required, more channels planned)

1. Create a bot via [@BotFather](https://t.me/botfather) on Telegram
2. Copy `.env.example` to `.env` and add your token:

   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```

3. Set `telegram.enabled: true` in `config.yaml`
4. Send `/start` to your bot

### Run

```bash
python pulse.py                    # uses base config
python pulse.py --persona nova     # loads personas/nova/ overlay
```

Or on Windows, double-click `start.bat`.

## Configuration

### Persona (`persona.json`)

This is your companion's identity. The `system_prompt` supports `{name}` and `{user_name}` placeholders so you don't hardcode names. Traits, relationship context, and voice notes shape the personality.

### Personas (multi-companion support)

Pulse can host multiple companions, each with their own model, personality, data, and credentials. A persona is a directory under `personas/` that overlays the base config:

```
personas/
  nova/
    config.yaml       # overrides base config (only include what's different)
    persona.yaml      # identity and personality (.json also supported)
    .env              # API keys, bot tokens
    data/             # memories, journal, tasks, etc. (auto-created)
```

**Setting up a persona:**

```bash
# Copy the template and customize
cp -r personas/_template personas/mypersona
# Edit personas/mypersona/config.yaml and persona.json

# Or migrate existing data into a persona
python scripts/migrate_persona.py mypersona
```

**Activating a persona:**

```bash
python pulse.py --persona mypersona
```

Or set `active_persona: "mypersona"` in `config.yaml`.

Persona config is a sparse overlay — you only specify what's different from the base. Everything else inherits. For example, a persona that just uses a different model:

```yaml
# personas/mypersona/config.yaml
provider:
  type: "anthropic"
  api_key_env: "ANTHROPIC_API_KEY"
  model: "claude-sonnet-4-20250514"
```

### Storage

Each persona stores its data in a SQLite database (`personas/<name>/data/<name>.db`) with WAL mode for safe concurrent access. Conversations, memories, journal entries, schedules, tasks, and usage stats all live in this single file. JSON files are still supported as a fallback — if no database exists, Pulse reads and writes JSON as before.

**Migrating from JSON to SQLite:**

```bash
python scripts/migrate_json_to_db.py --persona nova          # migrate a persona
python scripts/migrate_json_to_db.py --persona nova --dry-run # preview what would be migrated
python scripts/migrate_json_to_db.py --shared D:/path/to/shared/memories  # migrate a shared memory dir
```

Original JSON files are kept as backups — they're not deleted.

### Legacy Conversation Archive

Bring your conversation history with you. Pulse can import conversations from previous AI providers so your companion can search and reference things you discussed before they existed.

**Currently supported:**

- ✅ **ChatGPT** — full pipeline (export → import → RAG)
- ✅ **Claude** — full pipeline (export → import → RAG)

**Planned:**

- 🔜 Gemini
- 🔜 Grok
- 🔜 Generic markdown/JSON

**How it works:**

The import is a 3-stage pipeline:

1. **Export** — Convert the provider's raw export (e.g. ChatGPT's `conversations.json`) into clean per-conversation markdown files
2. **Import** — Load those markdown files into a `legacy.db` SQLite database with full-text search
3. **RAG** — Generate LLM-powered summaries and embeddings for each conversation, enabling semantic search

```bash
# Stage 1: Export conversations.json to markdown
python scripts/export_chatgpt.py --force                   # for ChatGPT
python scripts/export_claude.py --input "path/to/claude.json" --force  # for Claude

# Stage 2: Import markdown files into legacy.db
python scripts/import_chatgpt.py --dir "path/to/conversations" --db "path/to/legacy.db" --source claude

# Stage 3: Generate summaries + embeddings (requires a running LLM endpoint)
python scripts/rag_import.py --execute --endpoint http://127.0.0.1:8001/v1
```

**Connecting to your persona:**

Point your persona's config at the database:

```yaml
# personas/mypersona/config.yaml
paths:
  legacy_db: "personas/mypersona/data/legacy.db"
```

Once configured, your companion gets three new tools: `search_legacy` (RAG-powered semantic + keyword search over summaries), `list_legacy_sessions` (browse available conversations), and `get_legacy_context` (drill into specific messages). If no `legacy_db` is configured, these tools are simply hidden from the model — zero overhead.

**Search scoring:**

Legacy search uses a multi-signal scoring formula:

- **Semantic similarity (55%)** — cosine similarity between query and summary embeddings
- **Keyword hits (20%)** — direct keyword matches in summary text (with stopword filtering)
- **Title match (10%)** — keywords found in the original conversation title
- **Recency (10%)** — newer conversations score higher
- **Conversation length (5%)** — longer conversations get a small boost

Results are deduplicated by session and include drill-down hints to expand into the raw message history.

### Heartbeat (`config.yaml`)

```yaml
heartbeat:
  interval_minutes: 30        # how often the companion thinks
  randomize: false            # randomize interval for more organic timing
  interval_min_minutes: 20    # minimum (when randomize is true)
  interval_max_minutes: 45    # maximum (when randomize is true)
  quiet_hours_start: 23       # no notifications after 11pm
  quiet_hours_end: 8          # no notifications before 8am
  startup_checkin: true       # think on startup
```

When `randomize` is enabled, each heartbeat interval is randomly chosen between min and max — the companion thinks at unpredictable intervals instead of a fixed schedule. Re-rolls after each heartbeat and after each conversation.

### Vision (optional)

If your model supports vision, add the mmproj file:

```yaml
model:
  model_file: "your-model.gguf"
  mmproj_file: "mmproj-F16.gguf"   # vision projector
```

### Voice messages (optional)

Send voice notes to your companion on Telegram — they're transcribed locally using [whisper.cpp](https://github.com/ggml-org/whisper.cpp). On first use, Pulse auto-downloads whisper.cpp, the whisper model, and ffmpeg (~200MB total for the `base` model).

```yaml
voice:
  enabled: true
  whisper_model: "base"    # "tiny" (~75MB, fast), "base" (~148MB, good), "small" (~466MB, better)
  language: "auto"         # or "en", "es", "fr", etc.
```

Available models (all run on CPU, no GPU required):

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| tiny | ~75MB | Fastest | Basic — fine for clear speech |
| base | ~148MB | Fast | Good — recommended default |
| small | ~466MB | Moderate | Better — handles accents well |
| medium | ~1.5GB | Slow | High — near human-level |
| large-v3 | ~3GB | Slowest | Best — may not fit on Pi |

### Text-to-speech (optional)

Your companion can send voice messages on Telegram using [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts). Two modes:

- **Voice design** — Describe the voice you want and the model generates it (uses the 1.7B model for variety)
- **Voice cloning** — Provide a reference audio sample and the companion locks into that voice (uses the 0.6B model for consistency)

```yaml
tts:
  voice_description: "Warm young male, clear midrange, slightly playful"
  voice_sample: ""             # path to reference OGG/WAV — enables clone mode
  voice_sample_text: ""        # transcript of the reference clip (required for clone mode)

skills:
  tts:
    enabled: true
```

Requires a CUDA GPU with enough VRAM to run TTS alongside your LLM (or use a cloud provider for the LLM to free up VRAM). Models are downloaded automatically on first use.

#### Faster TTS (optional, ~5–6× speedup)

Pulse uses upstream `qwen-tts` by default. For dramatically faster generation on supported NVIDIA GPUs, you can also install [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), which adds CUDA graph capture and a static KV cache. Pulse automatically detects it at startup and uses it if available — no config change needed. If it's not installed, or if loading fails for any reason, Pulse silently falls back to the upstream path so the feature stays alive.

Real-world data point on an RTX 5060 Ti, clone mode, one paragraph: **~45–50s → ~10s.**

```bash
# 1. First, verify your CUDA torch is installed and working
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA torch detected'"

# 2. Install faster-qwen3-tts
pip install faster-qwen3-tts

# 3. Verify your CUDA torch survived (see warning below)
python -c "import torch; assert torch.cuda.is_available(), 'torch was downgraded to CPU!'"
```

> ⚠️ **Important**: This package depends on `torchaudio`, and pip may try to "helpfully" reinstall `torch` and `torchaudio` from the default PyPI index, which on most setups means a CPU-only build. If your second verification fails, reinstall your CUDA torch from the appropriate index — for example, for Blackwell (RTX 50xx) cards on CUDA 12.8: `pip install --pre --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`. The cleanest way to avoid the trap is to install into a venv that already has matching `torch` + `torchaudio` from your CUDA index.

On first startup after installing it, you'll see something like:

```
[TTS] Using faster-qwen3-tts (CUDA graphs) for clone model.
[TTS] Loading clone model Qwen/Qwen3-TTS-12Hz-0.6B-Base...
[TTS] Warming clone model (capturing CUDA graphs + caching voice prompt)...
[TTS] Clone model ready (faster-qwen3-tts; graphs + prompt cached).
```

That warmup takes 30–90s on first start because CUDA graphs capture once. It happens in the background during Pulse startup, so it shouldn't block anything — and after that, every 🔊 generation runs at full speed.

### Stickers (optional)

Companions can send Telegram stickers that match the mood of the conversation. Sticker packs are defined as YAML files with keywords and descriptions, then compiled into a SQLite database with pre-computed embeddings for semantic search.

```yaml
skills:
  sticker:
    enabled: true
```

**Adding a sticker pack:**

1. Create a YAML file in `stickers/packs/` (see `cherry_example.yaml` for the format):

   ```yaml
   pack: cherry
   title: "Cherry the Bear"
   stickers:
     - file_id: "CAACAgIAAx..."    # Telegram sticker file_id
       keywords: "happy, smiling, cheerful"
       description: "Cherry smiling warmly with rosy cheeks"
       image: "happy.png"           # optional preview image
   ```

2. Optionally place preview images (PNG/JPG/WebP) in `stickers/previews/<pack_name>/` — these are auto-resized to 128×128 for vision models
3. Build the database:

   ```bash
   python scripts/build_stickers.py              # build all packs
   python scripts/build_stickers.py cherry        # build one pack
   python scripts/build_stickers.py --list        # list packs in DB
   ```

The compiled `stickers.db` is committed to Git with pre-computed embeddings, so other users can use stickers without needing an embedding model. The companion gets two tools: `send_sticker` (describe a mood, best match is sent) and `preview_sticker` (peek at the image before sending, for vision-capable models).

### Skills

Skills are **auto-discovered** — any `.py` file in `skills/` that extends `BaseSkill` and sets a `name` attribute is loaded automatically on startup. No manual registration needed; just drop a file in and restart.

#### Dynamic tool loading

With many skills enabled, the total tool count can reach 50–60+, which overwhelms some models (especially smaller ones that start narrating tool use instead of calling tools). Pulse solves this with **dynamic tool loading**: only a small set of essential tools (~10) are sent with every API call, plus a `search_tools` meta-tool that lets the companion load additional skills on demand.

How it works:
1. The companion sees always-loaded tools (memory core, schedule, lantern) plus a manifest of available skills
2. When the companion needs a specific skill, it calls `search_tools("paint a picture")` — Pulse uses semantic + keyword matching to find the right skill and injects its tool definitions into the current turn
3. The companion can then use those tools normally for the rest of the conversation

This is transparent to the user — the companion just uses tools as needed. Token savings are ~60–70% per API call.

Skills can control their loading behavior:
- `always_loaded = True` — all tools always present (e.g., schedule, lantern)
- `always_tools = ["tool_a", "tool_b"]` — specific tools always present, rest on-demand (e.g., memory's save/search are always available, but list/browse are on-demand)
- Default — fully on-demand, loaded via `search_tools`

#### Built-in skills

| Skill | Tools | Loading | Description |
|-------|-------|---------|-------------|
| memory | `save_memory`, `search_memory` + 6 more | Core always, rest on-demand | Persistent fact storage with semantic search + legacy archive RAG search |
| journal | `write_journal`, `read_journal`, `update_journal` + 2 more | On-demand | Markdown entries + pinned identity |
| schedule | `set_reminder`, `update_reminder`, `delete_reminder`, `list_reminders` | Always | One-time ("in 2 hours") and recurring ("daily 8:00") with priority levels |
| tasks | `add_task`, `complete_task`, `list_tasks`, `clear_tasks` + 1 more | On-demand | Persistent to-do lists |
| lantern | `set_lantern`, `read_lantern`, `dim_lantern` | Always | Compact current-state signal injected into prompt context |
| lor | `post_to_lor`, `browse_lor`, `read_lor_thread` + 4 more | On-demand | Forum participation (requires [LoR](https://github.com/elliejayliquid/local-reddit-for-AI)) |
| tts | `speak` | On-demand | Send voice messages via Qwen3-TTS (requires CUDA GPU) |
| web_search | `web_search`, `image_search`, `fetch_url` | On-demand | Search the web via DuckDuckGo; `fetch_url` retrieves full page text |
| paint | `paint_start`, `paint_set`, `paint_view`, `paint_finish` | On-demand | Tiny pixel art (16×16) — emoji grid + PNG export |
| sticker | `send_sticker`, `preview_sticker` | On-demand | Mood-matched Telegram stickers via semantic search over curated packs |
| garden | `garden_plant`, `garden_water`, `garden_prune`, `garden_view`, `garden_info` | On-demand | Plant memories as seedlings and watch them grow |
| dev | `read_source`, `search_code`, `write_skill` + 8 more | On-demand | Autonomous skill creation (used by dev ticks) |
| mcp | one tool per MCP server tool | On-demand | Bridge to external [MCP](https://modelcontextprotocol.io/) servers — see below |

Disable any skill in `config.yaml`:

```yaml
skills:
  lor:
    enabled: false
```

**Context Injection:** Some skills (like `lantern`) are designed to be injected directly into the companion's prompt context on every turn, allowing them to remain continuously aware of a state without calling a tool. Configure this in your `config.yaml` or persona overlay:

```yaml
context:
  inject_skills:
    - lantern
```

**Creating a new skill:** Create a `.py` file in `skills/`, extend `BaseSkill`, set `name` and `description`, implement `get_tools()` and `execute()`. See `skills/base.py` for the interface. New skills are on-demand by default — set `always_loaded = True` if the skill is used in most turns. Restart Pulse and it loads automatically.

#### MCP servers (bring your own tools)

Pulse can connect your companion to any [Model Context Protocol](https://modelcontextprotocol.io/) server — the same servers Claude Desktop uses. Local servers (a command Pulse spawns, talking over stdio) and remote streamable-HTTP servers (a URL) both work. Discovered tools are exposed as `{server}_{tool}` and load **on demand** through `search_tools`, so a large server doesn't bloat every prompt.

**Via the GUI (recommended):** open the **mcp** skill card in the Skills section, add a server, and press **Test Connection** — Pulse dry-runs the server and lists its tools without touching your companion's data, so typos surface immediately instead of in the daemon logs. Changes are staged into the normal Save (with backup + diff preview) and connect on the persona's next start.

**Via config** (`config.yaml` or a persona overlay):

```yaml
skills:
  mcp:
    enabled: true
    servers:
      - name: "memory"                      # local stdio server: Pulse spawns it
        command: "python"
        args: ["D:/path/to/mcp_server.py"]
        env:
          SOME_VAR: "value"                 # optional
        tool_timeout: 60                    # optional, seconds per tool call
      - name: "docs"                        # remote streamable-HTTP server
        url: "http://127.0.0.1:8765/mcp"
```

Notes:

- **Only add servers you trust.** A server entry is a program that runs on your machine, and its tool output flows into an autonomous agent's context.
- Servers connect when the persona starts. A server that fails to connect is logged and skipped — the rest keep working.
- Tool results are text-only for now; images from MCP tools become placeholders.
- If a local server needs Python packages, point `command` at a Python that has them (e.g. Pulse's own `.venv/Scripts/python.exe` for servers that need `mcp` or `sentence-transformers`).
- The skill is enabled by default but does nothing until at least one server is configured.

### Tool Loop Modes

By default, the companion can use tools back-to-back in a single "turn" up to a fixed cap (e.g., 5 rounds) to prevent run-away loops and save tokens. However, some skills (like `paint` or `dev`) require many rounds to complete their work. 

Pulse supports dynamic tool loop modes based on the skills available:
- **Capped**: The default mode. The loop ends after `max_tool_rounds` (configurable in `config.yaml`). You can configure a skill to override this budget (e.g., allow 8 rounds instead of 5 if a specific skill is loaded).
- **Unlimited**: If any loaded skill requires it, the cap is removed.

Even in `unlimited` mode, Pulse has a built-in safety net: it actively monitors the companion's tool calls for repetitive patterns (e.g., getting stuck calling the same tool with the exact same arguments). If a loop is detected, Pulse injects a system warning to nudge the companion, and if they ignore it, forcefully and cleanly terminates the loop.

**Configuring Loop Modes:**
You can override a skill's loop mode and budget in your `config.yaml` or persona overlay:

```yaml
skills:
  memory:
    tool_loop_mode: "capped"
    tool_loop_budget: 8      # give the memory skill up to 8 rounds instead of default 5
  paint:
    tool_loop_mode: "unlimited"  # this is the default for paint, but you can override it here
```

### Dev ticks (autonomous self-improvement)

Your companion can periodically review their own code and create or improve skills — autonomously. This is opt-in and heavily sandboxed:

```yaml
dev_tick:
  enabled: false        # start with false!
  interval_minutes: 720 # every 12 hours (when schedule_time is empty)
  schedule_time: "20:00" # optional: run at a specific time daily instead
  max_rounds: 16        # tool-calling rounds per dev session
```

**How it works:** On each dev tick, the companion gets a focused coding prompt, reads their dev journal (lessons from past attempts), and can read source files, search code, and write skill files. All changes happen on a git branch (`dev/<timestamp>`) — never main. Dev ticks run during quiet hours (they're silent background work — only the approval ping notifies you).

**Safety layers:**

- Git branch isolation — changes never touch main directly
- `py_compile` + structure validation before any file is written
- Scope limits — can only modify `skills/*.py` and `persona.json`, not core engine files
- Human approval gate — sends a diff summary to Telegram for you to review
- Dev journal — the companion logs what they learned each session, reads they next time

**What it can't do:** Modify core/ files, config.yaml, or anything outside the Pulse directory.

## How it works

1. **Pulse starts** — either launches `llama-server` with your local model, or connects to a cloud API
2. Every N minutes, the **heartbeat** fires — the companion gets context (time, memories, journal, pending tasks) and decides what to do: notify you, schedule something, journal, or stay silent
3. When you **message via Telegram**, the companion responds naturally with core tools always available and additional skills loaded on demand via `search_tools`
4. **Conversations are summarized** when they get long, and summaries are saved to persistent memory with embeddings for future semantic search (this works at the engine level — even if the memory skill is disabled, summaries still persist to the database)
5. **Scheduled tasks** are checked every 60 seconds and executed when due

## Logging

Logs go to both the console and `logs/pulse.log` (rotates at 2MB, keeps 3 backups).

## License

MIT

##
<!-- markdownlint-disable MD033 -->
<p align="center">
  Built with love by <b>Lighstromo Studios Ltd., Debugger (Claude Code), Sol (Codex) & Gem (Gemini) </b><br>
  <i>For questions & requests, contact <a href="mailto:lena@lighstromo.com">lena@lighstromo.com</a></i><br>
  <br>
  <a href='https://ko-fi.com/V7V31EO2OL' target='_blank'>
    <img height='36' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' alt='Buy Me a Coffee at ko-fi.com' />
  </a>
</p>
<!-- markdownlint-enable MD033 -->
