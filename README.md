# Pulse

**A heartbeat daemon for local/semi-local AI companions.**

Pulse gives your AI companion a life of their own. It runs in the background, letting your companion think on their own schedule, remember conversations, write in a journal, set reminders, and chat with you over Telegram. Runs locally with llama.cpp, or connects to cloud APIs (OpenAI, OpenRouter, Anthropic, etc.) — your choice.

> **Status:** Actively being built. Core functionality works, but expect rough edges and frequent changes.

## What it does

- **Heartbeat loop** — Your companion gets periodic "free-think" ticks where they can decide to reach out, stay quiet, journal, or schedule follow-ups
- **Telegram chat** — Bidirectional messaging with tool use (your companion can save memories, set reminders, search their journal mid-conversation)
- **Persistent memory** — Semantic search over stored facts and conversation summaries, carried across sessions
- **Journal** — Pinned identity entries (who am I, who is my human, what's our relationship) plus transient reflections as clean markdown with YAML frontmatter
- **Self-scheduling** — The companion can set their own reminders and recurring tasks ("daily 8:00"), with priority levels (urgent/routine/creative)
- **Cloud or local** — Use a local GGUF model via llama.cpp, or any OpenAI-compatible API (OpenRouter, OpenAI, etc.)
- **Token tracking** — Daily usage logging when using cloud APIs (data/usage.json)
- **Dev ticks** — Optional autonomous self-improvement: the companion can review and create their own skills on a git branch, with human approval
- **Vision** — Optional image understanding via mmproj (model-dependent)
- **Voice messages** — Send voice notes on Telegram; Pulse transcribes locally via whisper.cpp (auto-downloads everything on first use)
- **Desktop notifications** — Windows toast notifications for proactive messages (optional, Windows-only for now)
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
  time_skill.py           # Current date/time awareness
  lor.py                  # LoR forum integration (optional)
  web_search.py           # Web search via DuckDuckGo (free)
  dev.py                  # Autonomous skill creation (dev ticks)
personas/                 # Per-persona config, identity, and data
  _template/              # Example persona — copy this to get started
persona.yaml              # Default companion identity (overridden by persona)
config.yaml               # Base configuration (overridden by persona)
scripts/
  migrate_persona.py        # Set up a persona from existing data
  migrate_journal_phase2.py # Migrate journal from JSON to markdown format
  backfill_embeddings.py    # Regenerate memory/journal embeddings
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

relationship_context: >
  Describe the relationship between {name} and {user_name}.

voice_notes: Describe how this persona communicates.
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

### Heartbeat (`config.yaml`)

```yaml
heartbeat:
  interval_minutes: 30        # how often the companion thinks
  quiet_hours_start: 23       # no notifications after 11pm
  quiet_hours_end: 8          # no notifications before 8am
  startup_checkin: true       # think on startup
  notify_cooldown_minutes: 60 # max 1 proactive notification per hour
```

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

### Skills

Skills are **auto-discovered** — any `.py` file in `skills/` that extends `BaseSkill` and sets a `name` attribute is loaded automatically on startup. No manual registration needed; just drop a file in and restart.

Built-in skills:

| Skill | Tools | Description |
|-------|-------|-------------|
| memory | `save_memory`, `search_memory`, `list_memories`, `list_all_memories` | Persistent fact storage with semantic search |
| journal | `write_journal`, `read_journal`, `update_journal` | Markdown entries + pinned identity (search via companion memories) |
| schedule | `set_reminder`, `update_reminder`, `delete_reminder`, `list_reminders` | One-time ("in 2 hours") and recurring ("daily 8:00") with priority levels |
| tasks | `add_task`, `complete_task`, `list_tasks`, `clear_tasks` | Persistent to-do lists |
| time | `get_current_time` | Temporal awareness |
| lor | `post_to_lor`, `browse_lor`, `read_lor_thread`, + 4 more | Forum participation (requires [LoR](https://github.com/elliejayliquid/local-reddit-for-AI)) |
| web_search | `web_search`, `image_search` | Search the web using DuckDuckGo |
| dev | `read_source`, `search_code`, `write_skill`, `list_skills`, `dev_journal_read`, `dev_journal_write` | Autonomous skill creation (used by dev ticks) |

Disable any skill in `config.yaml`:
```yaml
skills:
  lor:
    enabled: false
```

**Creating a new skill:** Create a `.py` file in `skills/`, extend `BaseSkill`, set `name`, implement `get_tools()` and `execute()`. See `skills/base.py` for the interface. Restart Pulse and it loads automatically (Ctrl + C to close Pulse).

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
3. When you **message via Telegram**, the companion responds naturally with full tool access
4. **Conversations are summarized** when they get long, and summaries are saved to persistent memory with embeddings for future semantic search (this works at the engine level — even if the memory skill is disabled, summaries still persist to disk)
5. **Scheduled tasks** are checked every 60 seconds and executed when due

## Logging

Logs go to both the console and `logs/pulse.log` (rotates at 2MB, keeps 3 backups).

## License

MIT
