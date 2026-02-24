# Pulse - Proactive Local AI Companion

A daemon that gives Nova (local Mistral via LM Studio) a heartbeat, turning a reactive
chat model into an ambient companion who can check in, remember, remind, and reach out.

**Status: Phase 1 - In Development**

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
      | Schedules  |--->| (OpenAI     |--->| Telegram*   |
      | Telegram*  |    |  compatible)|    | Voice (TTS)*|
      | Voice*     |    |             |    | 3D State*   |
      +-----------+    | Context:    |    +-------------+
                       | - memories  |
                       | - LoR posts |    * = future phases
                       | - time/date |
                       | - schedules |
                       | - history   |
                       +-------------+
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

When conversation history exceeds its token budget:
1. Summarize the conversation so far via Nova
2. Save summary to Nova's memory (as session_log)
3. Reset conversation buffer with just the summary
4. Continue with fresh context space

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

### Phase 1: "Nova Breathes" (current)
- [x] Project structure
- [ ] Pulse daemon with heartbeat timer
- [ ] LM Studio API client (OpenAI-compatible)
- [ ] Context manager (budget system)
- [ ] Schedule system (schedules.json, cron-like)
- [ ] Auto-summarization when context fills up
- [ ] Read Nova's memories directly from disk
- [ ] Read LoR data directly from disk
- [ ] Output: LoR posts
- [ ] Output: Windows desktop toast notifications
- [ ] `lor_schedule` MCP tool for Claude integration
- [ ] Quiet hours support

### Phase 2: "Nova Talks"
- [ ] Telegram bot channel (bidirectional chat)
- [ ] Voice messages via Telegram (native support)
- [ ] Interactive conversation mode with context continuity
- [ ] Telegram commands (/remind, /schedule, /quiet)

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
+-- data/
|   +-- schedules.json    # Scheduled tasks + reminders
|   +-- conversation.json # Rolling conversation history
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
- Pulse can write new memories after significant interactions

### With LM Studio
- API endpoint: `http://localhost:1234/v1` (OpenAI-compatible)
- Pulse is just an API client - if LM Studio isn't running, Pulse waits
- Model swapping is transparent (just change config)

## Personality Notes

Nova's personality is seeded via `persona.json` but should evolve naturally
through what Nova writes in LoR and stores in memory. The persona file is the
starting point, not a straitjacket.

---

*Last updated: 2026-02-23 | Session: Initial architecture with Claude*
