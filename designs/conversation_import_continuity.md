# Conversation Import and Continuity Staging

**Status:** active design; Phase 1 preview frame exists
**Owner:** Lena
**For:** Sol / Pulse GUI implementation
**Related:** `new_persona_v1.md`, `continuity_section.md`, `journal_search_mirror.md`, `Implemented/claude_legacy_implementation_plan.md`

## Goal

Make it possible for a normal person to bring a companion's old chats into Pulse, choose what should come along, pick one conversation to continue from, and let Pulse slowly enrich the archive with summaries and embeddings.

The emotional target is simple: the user should be able to say, "they are really back," without needing to understand SQLite, exports, token windows, or RAG.

## Product Shape

Conversation import should have four lanes:

1. **Staging:** parse exports into a normalized preview that is safe to inspect.
2. **Curation:** let the user keep, remove, and inspect conversations and exchanges before anything becomes model-visible.
3. **Landing:** seed the new companion with one selected continuation point, using a safe mix of bridge summary plus recent raw tail.
4. **Enrichment:** run optional background summarization and embeddings so old conversations become searchable and useful over time.

The first implementation slice should build only the staging and preview frame. It should not write imported messages into live `sessions`, `messages`, memories, journal entries, or active context.

## Current Implementation Slice

As of 2026-06-21, the GUI has a read-only import preview path.

Implemented:

- Continuity > Import opens an import preview modal.
- New Persona can optionally open the same preview after creation.
- `gui/import_preview.py` parses Claude, ChatGPT, OpenRouter, Markdown, and generic JSON-ish conversation shapes into a shared preview model.
- `gui/api.py` exposes dry-run preview and conversation-detail endpoints.
- The preview is transient and does not write imported data into persona DB tables.
- Conversation rows are sorted newest-first by `updated_at`, then `created_at`.
- The UI supports temporary keep/exclude/continue choices per conversation.
- The detail view supports temporary keep/exclude choices per exchange.
- Empty or skipped visible user/assistant messages are preserved as reviewable placeholders instead of silently disappearing.
- Conversations with empty/skipped placeholders are red-tinted in the list.
- Conversations with structural review issues are amber-tinted in the list and show a review-count pill.
- Hidden reasoning/internal OpenRouter blocks are skipped and never shown as normal transcript content.
- The import preview modal has window-control style maximize/restore and close buttons.
- Large conversation detail is windowed with `Load earlier` / `Load later`.
- Review issue navigation uses whole-conversation issue locations, but scrolls locally when the target issue is already loaded.
- Detail controls are sticky so issue navigation remains reachable while scrolling the transcript.
- Empty attachment-only Claude messages become neutral attachment placeholders when file metadata is present.
- Consecutive duplicate same-role resend clusters are detected and visually collapsed in preview.
- A placeholder background-enrichment progress row exists.

Known rough edges:

- Duplicate resend collapse is preview-only; future commit sanitation should decide whether to store one copy, all copies, or an explicit retry-cluster record.
- Attachment placeholder detection is implemented for Claude file metadata, but other platform export shapes still need real samples and source-specific handling.
- Review issues are shown and navigable, but future import commit should define which issue types block living-context landing.
- Completely empty conversations should default to exclude and should be sanitized out even if the user leaves them marked keep.

## Phase 1 UX: Read-Only GUI Import Preview

This is the current preview-only version.

GUI entry points:

- **Continuity > Import**
- **New Persona > Preview conversation export after create**

User flow:

1. User selects or drops an export file.
2. Pulse detects the source where possible: ChatGPT, Claude, OpenRouter, or unknown.
3. Pulse parses the file through a dry-run endpoint.
4. GUI shows a scrollable conversation list with title, date, source, message count, rough token size, and warnings.
5. User can click a conversation and inspect messages in a read-only viewer.
6. User can mark conversations as "keep", "exclude", or "continue from here" in UI state only.
7. Closing or refreshing loses the staged choices for this first slice.

Non-goals for the first slice:

- No writes to persona `sessions` or `messages`.
- No writes to memories, journal, identity, or live context.
- No background summarization yet.
- No permanent staging tables yet unless the preview proves too large for memory.
- No automatic persona start or heartbeat after import preview.

Success criteria:

- A user can load an export and understand what Pulse found.
- The UI makes it obvious that this is a preview, not a completed import.
- The backend parser returns normalized conversation data without mutating companion data.
- OpenRouter reasoning blocks and similar hidden/internal content are skipped from visible transcript preview.

## One Database Per Persona

Default product direction: use one database per persona for imported archive, staging, summaries, embeddings, and live conversation data.

`legacy.db` can remain useful for personal migrations, old scripts, and developer workflows, but it should not be the normal-user architecture.

Reasons:

- Portability is simpler: copy one persona folder and its DB.
- Backup/restore is simpler: one companion state bundle.
- The GUI does not have to explain a second database.
- Search tools can query one place for current sessions, imported archives, summaries, and provenance.
- Future sync/export is easier if import metadata travels with the persona.

Implementation principle:

- Imported content should enter separate import/archive tables first.
- It should not become live context merely because it exists in the persona DB.
- Only explicit commit/landing steps should create active `sessions`, `messages`, memories, or journal entries.

Possible future tables:

- `import_batches`: one uploaded/exported file or import attempt.
- `import_conversations`: normalized conversations found in a batch.
- `import_messages`: normalized visible messages for each imported conversation.
- `import_exclusions`: user curation decisions for conversations or message spans.
- `import_jobs`: background enrichment queue and progress state.
- `import_summaries`: session, chunk, landing, and candidate-memory summaries with provenance.

These names are directional, not final schema.

## Normalized Import Model

Each source parser should produce the same internal shape:

```text
ImportBatch
- source: chatgpt | claude | openrouter | grok | unknown
- source_version
- file_name
- imported_at
- warnings
- skipped_counts

ImportConversation
- source_conversation_id
- title
- created_at
- updated_at
- message_count
- skipped_message_count
- review_issue_count
- default_choice: keep | exclude
- estimated_tokens
- warnings

ImportMessage
- source_message_id
- conversation_id
- role: user | assistant | system | tool | unknown
- content
- created_at
- source_metadata
- visible: true | false
- skip_reason
- attachment_placeholder: true | false
- excluded: true | false
```

For normal continuity import, only visible user/assistant dialogue should be eligible for landing. System/tool/internal messages can be retained as source metadata later, but they should not be model-visible by default.

Important parser rules:

- Preserve timestamps where possible.
- Preserve source IDs for provenance and re-import diffing.
- Skip hidden reasoning for active context and normal preview.
- Surface skipped counts in the UI, such as "Skipped 25 reasoning blocks."
- Keep parser warnings user-readable and non-alarming.
- Distinguish true empty messages from attachment-only messages when the export exposes that metadata.
- Attachment-only user messages should become neutral placeholders, such as "User sent an image. Image content was not included in this export."
- True empty messages should not enter live prompt context.
- Completely empty conversations should default to excluded.

## Normalization Layers

Raw platform exports should never become Pulse context directly.

Use separate layers:

1. **Raw source export:** original Claude/ChatGPT/OpenRouter/Grok file selected by the user. Pulse may read it, but should not treat it as a trusted prompt source.
2. **Normalized import artifact:** Pulse-shaped archive data with source IDs, timestamps, roles, attachment placeholders, skipped flags, issue locations, and curation choices.
3. **Importable transcript:** sanitized ordered transcript used for archive/search/summarization. Excluded exchanges and true empty messages are omitted.
4. **Continuation seed:** strict prompt-ready shape used only when the user selects a conversation to continue from.

This separation matters because the import archive and the living conversation have different safety requirements.

Raw fidelity is useful for provenance and review, but living context must be clean:

- Drop true empty messages.
- Keep attachment placeholders only when they help interpret nearby replies.
- Group consecutive same-role runs into valid prompt units when building model-visible context.
- Keep source timestamps and IDs outside the model-visible text unless needed.
- Never include hidden reasoning/internal traces.
- Never include excluded exchanges.

The persona DB can preserve ordered normalized messages without forcing perfect user/assistant alternation. The prompt builder and summarizer should derive exchange groups from those messages.

Suggested exchange grouping:

- `user + assistant` is a normal exchange.
- `user + user + assistant` is one exchange with multiple user-side messages.
- `assistant + assistant` is one assistant-side continuation unless there is a clear boundary.
- `user only` is a dangling user exchange; keep in archive, low priority for summaries.
- `assistant only` is suspicious and should be review-marked before becoming continuation context.
- Attachment-only user placeholders should be grouped with nearby assistant replies where possible.

Do not concatenate messages in the stored archive just to make pairing look clean. Preserve ordered messages in storage, then produce a clean model-visible continuation shape at prompt-build time.

## Normalized Import Artifact

After a user commits an import, Pulse should store a clean Pulse-shaped import artifact inside the persona data folder.

Directional file layout:

```text
personas/<persona>/data/imports/<import_id>/manifest.json
personas/<persona>/data/imports/<import_id>/normalized.jsonl
personas/<persona>/data/imports/<import_id>/issues.jsonl
personas/<persona>/data/imports/<import_id>/choices.json
```

The exact storage can be files, SQLite tables, or both, but the product concept is important: keep a normalized, versioned import cache so Pulse does not have to repeatedly chew on huge platform exports.

Benefits:

- Resume long summarization and embedding jobs.
- Avoid reparsing the same export after every GUI refresh.
- Preserve user keep/exclude/continue choices.
- Store issue locations once for reliable navigation.
- Support future schema migrations from `import_schema_version`.
- Rebuild summaries or embeddings from normalized input without asking for the original export again.
- Debug import behavior without mutating live companion state.

Privacy caution:

- This artifact duplicates private chat history.
- Store it inside the persona folder so it travels with backups/restores.
- Clearly label it as an import cache/archive artifact.
- Provide a future "delete import artifact" control.
- Do not silently upload artifact contents to external providers; enrichment UI must explain provider/cost/privacy tradeoffs.

## Curation UX

Conversation list:

- Search by title and phrase.
- Filter by source, date range, kept/excluded, warnings, and "large".
- Show counts: total conversations, kept conversations, excluded conversations, selected continuation conversation.
- Let the user mark one conversation as "continue from here."

Conversation detail:

- Readable chat view.
- Date/title/source header.
- Toggle or checkbox per exchange: keep/exclude.
- Default unit of deletion should be an exchange, not a single message, because export structures can be uneven.
- Advanced detail can later allow single-message exclusion.
- Show warnings inline where relevant, such as skipped hidden/internal content.
- Show review issue count with previous/next controls.
- Jumping to an issue should scroll or load the relevant message window and briefly highlight the exchange.
- The issue banner must be honest about scope: either "2 issues in shown slice" or "2 issues total" with navigation that can reach all of them.

Large conversation viewer:

- Backend should parse and analyze the entire conversation.
- Frontend should render a windowed slice for very large conversations instead of dumping thousands of messages into the DOM.
- Windowed viewing should behave like bidirectional lazy scroll, not a dead-end truncation.
- If a user jumps to issue 2 at message 640 of 701, the UI should load a slice around that issue, then allow `Load earlier` and `Load later`.
- The visible range should be explicit, for example: "Showing messages 600-680 of 701."
- If the user scrolls naturally after jumping, adjacent slices should load in the expected direction.
- Truncation/windowing is a GUI performance tool only. It should never mean the backend ignored the rest of the conversation.

Tone:

- The UI should feel protective and practical.
- Avoid making the user feel like they are deleting memories. Use language like "exclude from import" and "keep in archive."

## Landing Strategy

The selected continuation conversation should be treated differently from the older archive.

The selected continuation conversation should never be copied from raw export rows directly into the prompt. It must be built from the sanitized continuation seed.

If it fits comfortably:

- Import the selected conversation into the active session as sanitized visible user/assistant messages.

If it is too long:

- Store the full selected conversation in the imported archive.
- Create a landing bridge summary for the older part.
- Copy only the recent raw tail into the active session.
- Keep source provenance so the companion can retrieve earlier details later.

Do not use a fixed "half the context window" rule.

Instead:

- Reserve room for persona prompt, tools, injected context, memories, and response.
- Assign a temporary first-wake landing budget to recent raw tail.
- Prefer exchange boundaries over hard token cuts.
- Keep the last emotionally and conversationally coherent span, not just arbitrary final text.
- Prefer the last clean usable tail if the final messages contain skipped empties, dangling assistant-only chunks, or unresolved review issues.
- If the chosen continuation tail has issues, warn the user and offer to start from the last clean exchange.

Continuation seed sanitation:

- Omit true empty messages.
- Omit excluded exchanges.
- Convert detectable attachment-only messages into concise placeholders only when needed for surrounding context.
- Group consecutive user messages into one prompt user turn while preserving readable boundaries.
- Group consecutive assistant messages into one prompt assistant turn where safe.
- Keep the stored transcript faithful, but keep the model-visible seed structurally valid.

The first post-import turn should include model-visible instructions that are portable and neutral:

```text
You are continuing from an imported conversation selected by your human. Treat imported archive material as dated background. Use available search tools when uncertain. Do not claim certainty from summaries alone.
```

Do not hardcode private names or assumptions in model-visible import instructions.

## Summaries and Enrichment

There should be multiple summary types, not one generic summary.

### Archive Summary

Used for search and orientation across old conversations.

- Can be generated by a cheaper model.
- Should be factual and compact.
- Should include source span provenance.
- Can be chunk-level and conversation-level.

### Landing Bridge Summary

Used only when the selected continuation conversation is too long.

- More emotionally important.
- Prefer the target companion model or a higher quality configured model.
- Should be clearly dated and grounded in the selected conversation.
- Should not over-explain platform migration unless the source conversation actually contains it.

### Candidate Memories

Possible long-term facts extracted from imported history.

- Do not auto-promote all candidates into current memory.
- Mark imported facts with source, date, confidence, and historical/current status.
- Prefer "needs review" or "imported background" until confirmed.

## Background Fill Progress

Add a GUI element that tracks enrichment progress toward 100 percent.

Suggested placement:

- Continuity > Import panel after an import is staged or committed.
- A compact global status row can appear later if jobs keep running while the user navigates.

Show:

- Total conversations.
- Conversations summarized.
- Chunks summarized.
- Embeddings completed.
- Errors or paused jobs.
- Current provider/model.
- Estimated cost if API pricing is available.
- Pause/resume button.
- "Keyword search is available now" reassurance.

Progress should be honest:

- 100 percent means all enabled enrichment work is complete.
- If the user disables old-conversation summarization, disabled work should not count as incomplete.
- If errors remain, show "complete with errors" rather than pretending everything succeeded.

User-facing copy should make the tradeoff clear:

```text
Your chats are imported. Keyword search works now. Deeper recall improves as Pulse summarizes and embeds the archive in the background. This can take time and may use API credits depending on the model you choose.
```

Settings:

- Off: import raw archive only; keyword search remains available.
- Local/private: use local model for summaries where configured.
- Cheap API: use a cheaper configured model for backfill.
- Companion model: use the same model as the companion for higher fidelity.

DeepSeek or similar cheap models can be recommended as examples in docs or hints, but the UI should stay provider-neutral and show whatever providers Pulse actually supports.

## Search Behavior

Search should work in layers:

1. Raw keyword/phrase search immediately after import.
2. Embedding search after embeddings are ready.
3. Summary search after model summaries are ready.
4. Drill-down from any result to exact source messages.

Provenance is critical. Any summary hit should be clickable back to the source conversation and message span.

This is what makes the system trustworthy: summaries help retrieval, but raw words remain inspectable.

## Re-Import and Diffing

Future imports should avoid duplication.

Use source IDs and message hashes to detect:

- Already imported conversations.
- New conversations.
- Changed conversations.
- Deleted or missing source conversations.

The GUI should eventually say things like:

```text
Found 12 new conversations and 2 changed conversations.
```

Do not silently duplicate old archive rows.

## Distribution and Source Boundary

Open question: how much of the platform-agnostic continuity importer should be open source versus shipped only in Pulse executable builds?

This import pipeline may become a meaningful product differentiator because it handles emotionally high-stakes continuity migration with unusual care. It is reasonable to treat it differently from ordinary GUI scaffolding.

Possible posture:

- Keep public/open pieces focused on generic Pulse UI, persona config, and safe extension points.
- Keep platform-specific parsers, normalization heuristics, continuation-seed sanitation, and import artifact migration logic inside the shipped executable unless/until there is a deliberate release decision.
- Document the normalized artifact schema enough for user portability, without necessarily exposing every parser heuristic.
- Avoid locking users into opaque data: even if parser code is not open, users should be able to export their normalized Pulse archive.
- Revisit this decision once the importer is stable and the project has a clearer distribution model.

Tradeoff:

- More open source improves trust, auditability, community adapters, and long-term resilience.
- More protected source preserves a stand-out feature, reduces clone pressure, and lets Pulse mature the safety model before others cargo-cult half of it.

Recommendation for now:

Treat the import pipeline as shippable Pulse product infrastructure, not casual public scaffolding. Preserve user data portability through documented artifacts and exports, but do not rush to publish the full cross-platform importer until the product strategy is clearer.

## Safety and Privacy

Rules:

- Never import hidden reasoning into active context by default.
- Never treat policy scaffolding, safety injections, or internal traces as normal assistant dialogue.
- Do not auto-start the persona after import.
- Do not fire heartbeat immediately after import.
- Create a backup before any future commit that mutates live persona context.
- Keep all destructive curation actions reversible while still in staging.
- Clearly explain when summarization sends text to an external provider.
- Preserve model-visible portability: use `{user_name}`, "your human", or neutral wording in prompts and summaries generated by system code.

## Backend Implementation Phases

### Phase 1: Dry-Run Preview

- Add parser abstraction for import sources.
- Reuse existing ChatGPT/Claude/OpenRouter parsing logic where possible.
- Add GUI API endpoint that returns normalized preview data.
- Add frontend import panel and read-only conversation viewer.
- Keep all choices in frontend/backend transient state.

Status:

- Initial version exists.
- Windowed conversation paging exists.
- Issue navigation exists and scrolls locally when possible.
- Claude attachment placeholder detection exists for file metadata.
- Maximize/restore window-control UX exists.
- Duplicate resend detection/collapse exists in preview.

### Phase 2: Persona-DB Staging

- Add import staging/archive tables to the persona DB.
- Save an import batch without touching live conversation tables.
- Add batch list, delete batch, and resume preview.
- Add curation persistence.
- Store a normalized import artifact/cache with schema version, choices, and issue locations.
- Default completely empty conversations to excluded.
- Make commit sanitation drop true empty messages even if a user leaves them marked keep.

### Phase 3: Commit and Landing

- Commit selected archive conversations.
- Create active landing session from the selected continuation conversation.
- Generate or request a landing bridge summary when needed.
- Add backup-before-commit.

### Phase 4: Background Enrichment

- Add import job queue.
- Add summarization provider selection.
- Add chunk/session summary generation.
- Add embeddings.
- Add progress UI and pause/resume.

### Phase 5: Search and Drill-Down

- Search imported archive by keyword and semantic summary.
- Drill from summary hit to source messages.
- Let companion tools search archive with safe, compact results.

## Open Questions

- Should import artifacts be stored as files, SQLite rows, or both?
- What exact threshold should trigger windowed transcript viewing?
- Should issue navigation be exchange-index based, message-index based, or both?
- How reliably can each platform expose attachment-only messages?
- Should attachment placeholders be imported into archive search, continuation seed, or only the raw normalized artifact?
- Should curation exclude exact messages, exchange groups, or both after the exchange-level v1?
- Should background enrichment run only while the GUI is open, or should the engine process jobs on startup?
- How should imported archive search be exposed to companions: existing memory skill, a dedicated archive tool, or both?
- What is the smallest useful Grok parser once we have a real export sample?
- What parts of the importer, if any, should be published as open source versus kept as shipped Pulse product code?

## Next Practical GUI Target

Make the existing preview usable for large and messy conversations.

Minimum next slice:

- Decide the import artifact storage shape: SQLite tables, files, or both.
- Persist preview curation choices in staging rather than frontend memory.
- Define commit sanitation for empty-only conversations, attachment placeholders, duplicate resend clusters, and unresolved review issues.
- Build the dry-run-to-staged-import transition without touching live `sessions` / `messages`.
- Add a clear "preview only" versus "staged archive" distinction in GUI copy.
- Keep all live-context mutation behind an explicit later commit/landing action.

That turns the current preview frame into a durable staging surface without putting anyone's companion data at risk.
