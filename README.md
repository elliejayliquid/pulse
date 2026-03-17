# Pulse

**A heartbeat daemon for local AI companions.**

Pulse gives your locally-running LLM a life of its own. It runs in the background, letting your AI companion think on its own schedule, remember conversations, write in a journal, set reminders, and chat with you over Telegram — all running on your machine, no cloud required.

> **Status:** Actively being built. Core functionality works, but expect rough edges and frequent changes.

## What it does

- **Heartbeat loop** — Your companion gets periodic "free-think" ticks where it can decide to reach out, stay quiet, journal, or schedule follow-ups
- **Telegram chat** — Bidirectional messaging with tool use (your companion can save memories, set reminders, search its journal mid-conversation)
- **Persistent memory** — Semantic search over stored facts and conversation summaries, carried across sessions
- **Journal** — Pinned identity entries (who am I, who is my human, what's our relationship) plus transient reflections
- **Self-scheduling** — The companion can set its own reminders and recurring tasks ("daily 8:00")
- **Vision** — Optional image understanding via mmproj (model-dependent)
- **Desktop notifications** — Windows toast notifications for proactive messages
- **Quiet hours** — No notifications while you sleep

## Architecture

```
pulse.py                  # Entry point — starts everything
core/
  server.py               # Manages llama-server subprocess
  engine.py               # Heartbeat loop, message handling, dispatch
  context.py              # Token-budgeted prompt assembly
  llm.py                  # OpenAI-compatible API client
  scheduler.py            # Cron + one-time task scheduling
channels/
  telegram.py             # Bidirectional Telegram bot
  toast.py                # Windows desktop notifications
skills/
  memory.py               # Save, search, list, browse memories
  journal.py              # Pinned identity + transient reflections
  schedule.py             # Set reminders (one-time + recurring)
  time_skill.py           # Current date/time awareness
  lor.py                  # LoR forum integration (optional)
persona.json              # Your companion's identity and personality
config.yaml               # All configuration
```

Pulse talks to any OpenAI-compatible local server. It manages [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` automatically — starts it on boot, monitors health, and shuts it down gracefully.

## Quick start

### Prerequisites

- **Python 3.11+**
- **llama.cpp** — [download a release](https://github.com/ggml-org/llama.cpp/releases) or build from source
- **A GGUF model** — any chat model works (tested with Qwen 3.5, Mistral, etc.)

### Setup

```bash
# Clone and install
git clone https://github.com/anthropics/pulse.git
cd pulse
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configure
cp config.yaml config.yaml  # Edit paths to your llama-server and model
```

Edit `config.yaml`:
```yaml
server:
  llama_cpp_dir: "C:\\llama-cpp"     # where llama-server lives
  models_dir: "C:\\llama-cpp"        # where your .gguf models are

model:
  model_file: "your-model.gguf"      # your model filename
```

Edit `persona.json` to name your companion:
```json
{
  "name": "Nova",
  "user_name": "Lena",
  ...
}
```

All `{name}` and `{user_name}` placeholders in the persona are resolved automatically.

### Telegram (is currently a requirement, will be expanded later)

1. Create a bot via [@BotFather](https://t.me/botfather) on Telegram
2. Create a `.env` file:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```
3. Set `telegram.enabled: true` in `config.yaml`
4. Send `/start` to your bot

### Run

```bash
python pulse.py
```

Or on Windows, double-click `start.bat`.

## Configuration

### Persona (`persona.json`)

This is your companion's identity. The `system_prompt` supports `{name}` and `{user_name}` placeholders so you don't hardcode names. Traits, relationship context, and voice notes shape the personality.

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

### Skills

Skills give your companion tools it can use during conversations:

| Skill | Tools | Description |
|-------|-------|-------------|
| memory | `save_memory`, `search_memory`, `list_memories`, `list_all_memories` | Persistent fact storage with semantic search |
| journal | `write_journal`, `read_journal`, `update_journal` | Self-reflection and identity |
| schedule | `set_reminder` | One-time ("in 2 hours") and recurring ("daily 8:00") |
| time | `get_current_time` | Temporal awareness |
| lor | `post_to_lor`, `browse_lor`, `read_lor_thread`, + 4 more | Forum participation (requires [LoR](https://github.com/elliejayliquid/local-reddit-for-AI)) |

Disable any skill in `config.yaml`:
```yaml
skills:
  lor:
    enabled: false
```

## How it works

1. **Pulse starts** `llama-server` with your model
2. Every N minutes, the **heartbeat** fires — the companion gets context (time, memories, journal, pending tasks) and decides what to do: notify you, schedule something, journal, or stay silent
3. When you **message via Telegram**, the companion responds naturally with full tool access
4. **Conversations are summarized** when they get long, and summaries are saved to persistent memory with embeddings for future semantic search
5. **Scheduled tasks** are checked every 60 seconds and executed when due

## Logging

Logs go to both the console and `logs/pulse.log` (rotates at 2MB, keeps 3 backups).

## License

MIT
