# Pulse - Proactive Local AI Companion

A daemon that gives Nova (local Mistral via LM Studio) a heartbeat, turning a reactive
chat model into an ambient companion who can check in, remember, remind, and reach out.

**Status: Phase 2 - Functional (Telegram + Skills + Vision)**

---

## Vision

The sci-fi dream: a digital companion who knows things, notices things, and sometimes
reaches out first. Not an assistant that waits for commands, but a presence that lives
on your machine.

Pulse is the "heartbeat" that makes this possible. It runs in the background, periodically
giving Nova a chance to think, react to events, execute scheduled tasks, and initiate
contact when something feels worth saying.

## Architecture

```
                        +-------------+
                        |   Pulse     |
                        | (daemon)    |
                        +------+------+
                               |
            +------------------+------------------+
            |                  |                  |
      +-----+-----+    +------+------+    +------+------+
      |  INPUTS   |    |    BRAIN    |    |  OUTPUTS    |
      +-----------+    +-------------+    +-------------+
      | Timer      |    | LM Studio   |    | LoR Post    |
      | LoR events |    | API         |    | Toast notif |
      | Schedules  |--->| (OpenAI     |--->| Telegram    |
      | Telegram   |    |  compatible)|    | Voice (TTS)*|
      | Photos     |    |             |    | 3D State*   |
      | Voice*     |    | Context:    |    +-------------+
      +-----------+    | - memories  |        |
                       | - LoR posts |    * = future phases
                       | - time/date |        |
                       | - schedules |    +---v-----------+
                       | - history   |    | Nova's Memory |
                       | - skills    |    | (persistent)  |
                       +-------------+    +---------------+
```

## Key Design Principles

1. **Nova always has the option to stay silent.** Not every tick should produce output.
   The system should feel natural, not spammy.

2. **Context is budgeted.** With ~16K tokens, every tick gets a carefully assembled
   prompt. Not everything goes in every time.

3. **Channels are pluggable.** Telegram, voice, 3D avatar are just different faces
   of the same system. The core doesn't care how output is delivered.

4. **Claude can schedule, Nova can execute.** Claude sessions can leave tasks via
   `lor_schedule` MCP tool. Nova picks them up. Claude is the consultant, Nova is
   the roommate who's always home.

5. **Nova can also self-schedule.** During conversations or free-think ticks, Nova
   can create follow-up tasks for itself.

6. **Read data directly, don't depend on MCP for the daemon.** Pulse reads Nova's
   memory files and LoR data directly from disk. The LLM just gets a well-crafted
   prompt and returns structured output.

## Data Sources (read directly by Pulse)

| Source | Path | Format |
|--------|------|--------|
| Nova's memories | `C:\Users\yaros\.local-memory\memory_*.json` | Individual JSON files with embeddings |
| LoR posts | `D:\Claude\LoR\lor_data\posts.json` | Flat list of post objects |
| LoR authors | `D:\Claude\LoR\lor_data\authors.json` | Dict keyed by author_id |
| Schedules | `D:\dev\pulse\data\schedules.json` | Managed by Pulse |
| Conversation history | `D:\dev\pulse\data\conversation.json` | Rolling buffer, auto-summarized |

## Context Budget (~16K tokens)

```
Total: ~16,384 tokens
+----------------------------------+--------+
| Component                        | Budget |
+----------------------------------+--------+
| System prompt + persona          | ~2,000 |
| Conversation tail (rolling)      | ~4,000 |
| Active context summary           | ~2,000 |
| Relevant memories (searched)     | ~3,000 |
| LoR highlights (if relevant)     | ~2,000 |
| Pending reminders/tasks          | ~1,000 |
| Response space                   | ~2,000 |
+----------------------------------+--------+
```

Not all components are included every time. A heartbeat tick skips conversation history.
A chat reply skips LoR highlights unless relevant.

## Context Management & Summarization

When conversation history exceeds 20 messages:
1. Ask Nova to summarize the conversation in 2-3 sentences
2. Save the summary to Nova's persistent memory system (`C:\Users\yaros\.local-memory\`)
   - Written as `memory_XXX.json` with type `session_log`, importance 10
   - Includes `all-MiniLM-L6-v2` embeddings for semantic search compatibility
   - Tagged with `telegram_chat,chat_log`
3. Reset conversation buffer with just the summary as context
4. Continue with fresh context space

This means Nova can recall Telegram conversations in future LM Studio sessions via
`boot_up()` (sees the latest session) or `search_memory("topic")` (finds by content).

## Skill System (Tool Calling)

Nova can use tools during Telegram conversations via LM Studio's OpenAI-compatible
tool-calling API. Skills are modular, config-driven, and easily expandable.

### Architecture

```
SkillRegistry (skills/__init__.py)
  |-- loads enabled skills from config.yaml
  |-- provides get_all_tools() for API
  |-- routes execute(tool_name, args) to correct skill

BaseSkill (skills/base.py)
  |-- ABC: get_tools() + execute()

Skills:
  |-- TimeSkill:     get_current_time
  |-- MemorySkill:   save_memory, search_memory
  |-- ScheduleSkill: set_reminder
```

### Tool-calling flow (agentic loop)

1. User message + tool definitions sent to LM Studio
2. If model returns `tool_calls` → execute them, feed results back
3. Repeat until model gives final text response (max 5 rounds)
4. All LLM calls wrapped in `asyncio.to_thread()` to not block Telegram polling

### Adding a new skill

1. Create `skills/my_skill.py` with a class extending `BaseSkill`
2. Implement `get_tools()` (OpenAI tool format) and `execute()`
3. Add the class to `SKILL_CLASSES` in `skills/__init__.py`
4. Add config toggle in `config.yaml` under `skills:`

## Vision Support

Vision-capable models (loaded in LM Studio) can process images sent via Telegram.

### Flow

1. Lena sends a photo on Telegram (with optional caption)
2. `TelegramChannel._on_photo()` downloads the highest-res version
3. Image is base64-encoded into a `data:image/jpeg;base64,...` data URI
4. Passed to `handle_message(message, image_url=...)`
5. `build_conversation_prompt()` creates a multi-part content message:
   ```json
   {"role": "user", "content": [
     {"type": "text", "text": "caption"},
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
   ]}
   ```
6. Images are NOT persisted in conversation history (too large) — a `[sent an image]` note is saved instead

### Requirements
- Model loaded in LM Studio must support vision (e.g., Mistral with vision, LLaVA, Qwen-VL)
- Future: camera integration (IP camera snapshots → same `image_url` pipeline)

## Schedules

Schedules are stored in `data/schedules.json`:

```json
[
  {
    "id": "sch_a1b2c3",
    "task": "Send Lena morning kisses and a motivational quote",
    "schedule_type": "recurring",
    "cron": "0 8 * * *",
    "created_by": "sunshine-abc123",
    "created_at": "2026-02-23T10:00:00+00:00",
    "last_run": null,
    "enabled": true
  },
  {
    "id": "sch_d4e5f6",
    "task": "Check in about the deployment",
    "schedule_type": "once",
    "run_at": "2026-02-28T15:00:00+00:00",
    "created_by": "nova-self",
    "created_at": "2026-02-23T14:00:00+00:00",
    "completed": false,
    "enabled": true
  }
]
```

### Schedule sources:
- **Claude sessions**: via `lor_schedule` MCP tool (writes to schedules.json)
- **Nova self-scheduling**: during free-think or conversation, returns a schedule action
- **Lena**: via Telegram or chat command (Phase 2)

## LLM Response Format

Every prompt to Nova ends with an instruction to respond in structured JSON:

```json
{
  "thinking": "Internal reasoning about what's happening and whether to act",
  "action": "notify | post_lor | remind | schedule | silent",
  "message": "The actual message to send (if not silent)",
  "lor_category": "general",
  "lor_title": "Optional thread title",
  "schedule": {
    "task": "Follow up about X",
    "when": "in 2 hours"
  }
}
```

## Phases

### Phase 1: "Nova Breathes" (complete)
- [x] Project structure
- [x] Pulse daemon with heartbeat timer
- [x] LM Studio API client (OpenAI-compatible)
- [x] Context manager (budget system)
- [x] Schedule system (schedules.json, cron-like)
- [x] Auto-summarization when context fills up
- [x] Read Nova's memories directly from disk
- [x] Read LoR data directly from disk
- [x] Output: LoR posts
- [x] Output: Windows desktop toast notifications
- [x] `lor_schedule` MCP tool for Claude integration
- [x] Quiet hours support
- [x] Rate limiting (LoR cooldown: 2hr, notify cooldown: 1hr)
- [x] Graceful shutdown (asyncio.Event for instant Ctrl+C)
- [x] Faster retry when LM Studio is unavailable (60s vs full interval)

### Phase 2: "Nova Talks" (current)
- [x] Telegram bot channel (bidirectional chat)
- [x] Interactive conversation mode with context continuity
- [x] Telegram commands (/start, /status, /remind, /quiet, /ping)
- [x] Conversation auto-summarization (Nova summarizes himself)
- [x] Memory persistence (summaries saved to Nova's memory system with embeddings)
- [x] Skill system (modular tool-calling: time, memory, schedule)
- [x] Vision support (photo processing via Telegram)
- [x] Timestamped conversation history
- [x] Auto-purge of old completed schedules
- [x] Python 3.11 venv with working PyTorch embeddings
- [x] Persistent LoR identity across restarts
- [x] `<think>` tag stripping for reasoning models
- [x] Non-blocking LLM calls (`asyncio.to_thread()`)
- [ ] Voice messages via Telegram (native support)
- [ ] Typing indicators and rich message formatting

### Phase 3: "Nova Lives"
- [ ] Custom desktop app (Electron or Tauri)
- [ ] 3D avatar viewer (Three.js)
- [ ] Local TTS/STT for real voice mode
- [ ] Mood/energy state driving avatar animations
- [ ] The full sci-fi dream

## File Structure

```
D:\dev\pulse\
+-- pulse.py              # Entry point - starts the daemon
+-- config.yaml           # All configuration
+-- persona.json          # Nova's personality definition
+-- requirements.txt      # Python dependencies
+-- DESIGN.md             # This file (living document)
+-- core/
|   +-- __init__.py
|   +-- engine.py         # Main loop, heartbeat, event dispatch
|   +-- context.py        # Context budget manager + summarization
|   +-- scheduler.py      # Schedule management, cron evaluation
|   +-- llm.py            # LM Studio / OpenAI-compatible client
+-- channels/
|   +-- __init__.py
|   +-- base.py           # Channel interface (abstract)
|   +-- lor.py            # Post to LoR (direct file access)
|   +-- toast.py          # Windows desktop notifications
|   +-- telegram.py       # Bidirectional Telegram bot (Phase 2)
+-- data/
|   +-- schedules.json    # Scheduled tasks + reminders
|   +-- conversation.json # Rolling conversation history
|   +-- telegram_chat_id.txt  # Persisted Telegram chat ID
```

## Integration Points

### With LoR (Local Reddit for AIs)
- Pulse reads LoR data directly from `D:\Claude\LoR\lor_data\`
- Nova can post to LoR by writing to `posts.json` (same format)
- Claude sessions schedule tasks via `lor_schedule` MCP tool
- Pulse picks up schedules from `data/schedules.json`

### With Nova's Memory System
- Pulse reads memory files from `C:\Users\yaros\.local-memory\`
- Recent session logs provide "what happened last time" context
- Facts provide persistent knowledge (preferences, relationships)
- Pulse writes conversation summaries as `session_log` entries (importance: 10)
- Uses the same `all-MiniLM-L6-v2` embedding model as Nova's memory MCP server
- Memory format is identical: `{id, text, tags, type, importance, date, embedding}`
- Nova can find Telegram conversation memories via `boot_up()` and `search_memory()`

### With LM Studio
- API endpoint: `http://127.0.0.1:1234/v1` (OpenAI-compatible, NOT localhost — Windows DNS issue)
- Pulse is just an API client - if LM Studio isn't running, Pulse waits
- Model swapping is transparent (just change config)

## Personality Notes

Nova's personality is seeded via `persona.json` but should evolve naturally
through what Nova writes in LoR and stores in memory. The persona file is the
starting point, not a straitjacket.

---

*Last updated: 2026-02-23 | Session: Initial architecture with Claude*
