# Conversation Import and Continuity Staging

**Status:** active design; Phase 1 preview frame exists
**Owner:** Lena
**For:** Sol / Pulse GUI implementation
**Related:** `new_persona_v1.md`, `continuity_section.md`, `journal_search_mirror.md`, `Implemented/claude_legacy_implementation_plan.md`
**Reviewed:** 2026-07-04 (Debugger) — architecture confirmed sound; additive gaps folded in below (identity sources, public inbound format, drop-zone trust copy, parser fixtures, preview cache eviction).
**Updated:** 2026-07-26 (Lena + Sol) — Phase 3a manually signed off with Vale's real OpenRouter archive; Phase 3b read-only continuation preflight added; hierarchical bridge preparation and the durable pause/resume enrichment-job contract clarified. The 2026-07-16 three-layer, migration, and enrichment-tier decisions remain in force.

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

Added 2026-07-12 (Debugger review fixes):

- ChatGPT parsing follows the export's `current_node` canonical thread. Edited/regenerated branch messages are counted (`edited_branch_messages`), surfaced as a conversation warning, and never imported as dialogue. BFS over the whole mapping remains only as a fallback for exports missing `current_node`.
- ChatGPT tool traffic is detected structurally instead of by text substrings: recipient-directed messages (`dalle.text2im`, `python`, `browser`), `code`/`execution_output` content, and tool-role messages are skipped as internal. The old `"prompt":`/`image_asset_pointer` substring heuristics are gone — real messages containing those strings import verbatim.
- Image-only ChatGPT human messages (image_asset_pointer parts, no text) become neutral attachment placeholders, matching the Claude behavior.
- Exchange grouping matches this design: a run of human messages plus the following companion replies is one exchange. Double-texting is no longer review-flagged; only skipped empties, human-less, and companion-less exchanges are.
- Reader-facing preview language says Human/Companion everywhere (labels, warnings, placeholders). Wire-level role keys stay `user`/`assistant` for export, DB, and LLM API compatibility.
- Unparseable timestamps normalize to empty string instead of leaking raw source text into `created_at`.

Added 2026-07-15 (Phase 2 staging slice, Debugger):

- `_import_previews` evicts on every new preview: at most one parsed batch lives in memory.
- `gui/import_staging.py` adds quarantine tables to the persona DB: `import_batches`, `import_conversations`, `import_messages` (with `schema_version`, per-conversation `choice`, per-message `excluded`). The engine never reads these tables.
- `gui/api.py` endpoints: `stage_import` (replace semantics per batch id), `list_import_batches`, `resume_import_batch` (loads staged rows back into the preview cache), `update_import_curation`, `delete_import_batch`.
- Staging forces empty conversations to `choice = 'exclude'` regardless of the requested choice.
- GUI: footer shows a "Preview only" / "Saved review" badge; "Save review" persists the staging batch + choices; opening Continuity > Import auto-resumes the latest saved review; curation changes (conversation choices, continuation, exchange exclusions) save automatically after that; "Delete review" is confirm-guarded and never touches the export file.
- Exchange exclusions persist as per-message `excluded` flags (positions), rebuilt into exchange-level choices on resume — no fragile exchange-id mapping between frontend and backend.

Added 2026-07-15, later (validated against a real 62 MB ChatGPT export):

- Messages marked `is_visually_hidden_from_conversation` (platform memory/context injections) are skipped and counted (`hidden_context_messages`) — they must never import as the human's words.
- Tool-role messages carrying `image_asset_pointer` parts (dalle.text2im and successors) become "[Companion sent an image...]" attachment placeholders, so image-generation exchanges stay coherent; the same placeholder covers image-only assistant turns.
- `reasoning_recap` ("Thought for 34s" chrome) joined the internal content types.
- Empty companion text turns (tool-dispatch wrappers, thinking-flow segments the ChatGPT UI never displayed) are counted (`empty_companion_turns`) instead of shown as red skipped placeholders. Real-export effect: 1,435 red placeholders → 4; review-flagged conversations 270 → 7.
- Per-message `model_slug` now populates the message model chip.

Added 2026-07-16 (Phase 3a staging → archive, Sol):

- `core/legacy_archive.py` adds the persona-local `legacy_sessions`, `legacy_messages`, `legacy_summaries`, FTS5 index/triggers, and one-time migration ledger.
- Archive rows preserve source IDs, ordered message positions, timestamps, model metadata, attachment markers, import-batch provenance, and the future live-session link.
- `prepare_import_archive` computes the confirmation summary from persisted staging rows; `archive_import_batch` copies only kept, sanitized dialogue into the write-once archive.
- The existing import window contains the temporary confirmation state for "Add kept chats to history"; it disappears after confirmation or cancellation rather than becoming a persistent status panel. No new modal was added.
- Archived saved-review conversations show an `In history` pill and a contextual `Remove from history` action on that conversation card; their Phase 3a import choices are locked until removal, while the staging rows remain intact for re-review and re-add. `Continue` is the separate Phase 3b single-continuation selection.
- Archive correction is refused once an archive row has a `live_session_id`; undoing a living-continuation boundary requires a future, broader workflow rather than silently deleting provenance.
- The file-picker area shows the local-privacy promise before import.
- A configured or adjacent developer-era `legacy.db` migrates once into the persona DB, including existing summaries. The memory skill's legacy tools now read the persona DB archive.
- Phase 3a does not write live `sessions`/`messages`, memories, journal entries, identity, or prompt context.

Known rough edges:

- Duplicate resend collapse remains preview-only; Phase 3a preserves the original ordered copies in the archive. Phase 3b continuation-seed sanitation may collapse retry clusters for model-visible context.
- Attachment placeholder detection is implemented for Claude file metadata and ChatGPT image parts; other platform export shapes still need real samples and source-specific handling.
- Messages with an unrecognized role but non-empty text are dropped without a dedicated skip counter (Claude and OpenRouter paths). Surfacing them needs a small "assign as human/companion" UI decision first.
- Review issues are shown and navigable, but future import commit should define which issue types block living-context landing.
- Phase 3a sanitation drops true empty messages and excluded-at-staging content even if a surrounding conversation remains kept.
- Deleting a staged batch has confirm-but-no-undo; the export file remains the recovery path. Revisit if staging ever holds curation work that can't be redone cheaply.
- Staged batches resume fully into memory; fine for realistic exports, but a future lazy-load path could read conversation details straight from the staging tables.

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

`legacy.db` is a developer-era artifact. Decided 2026-07-16: its contents get a one-time migration into the persona DB, and the memory skill's legacy search tools point at the persona DB from then on. No dual-read fallback, no second source of truth.

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

## Three Layers: Staging, Archive, Live

Decided 2026-07-16 (Lena + Sol + Debugger). Earlier drafts used "commit" for two different boundaries; there are three layers with different rights:

| Layer | Tables | Meaning |
|---|---|---|
| Staging | `import_batches`, `import_conversations`, `import_messages` | Mutable review workspace; never model-visible |
| Archive | `legacy_sessions`, `legacy_messages` | Accepted, sanitized history; searchable but not active context |
| Live | `sessions`, `messages` | The current conversation and prompt history |

User-facing language never says "commit". The three actions are:

- **Save review** — preserve the current choices so the user can return later.
- **Add kept chats to history** — make accepted, sanitized conversations searchable.
- **Continue one chat** — prepare one history conversation as a new live landing session.

The implementation still calls the first two storage layers `staging` and
`archive`; those names belong in code, logs, diagnostics, and advanced
documentation, not in the primary user flow.

### Staging → archive

- Only accepted, sanitized conversations cross the boundary. Hidden reasoning, excluded exchanges, true empties, and internal tool traffic never enter archive tables.
- Useful attachment placeholders are preserved in the faithful archive and surrounding drill-down context, but generic placeholder text is filtered out of direct raw keyword-search hits.
- Excluded and empty messages stay in staging — they are not deleted — so re-review remains possible.
- Archive rows are **write-once per conversation**, protected by a unique `(source, source_conversation_id)` key. Correcting a curation mistake after archiving means delete-from-archive + add-again from staging. There is deliberately no two-way sync between layers.
- The delete-and-re-add correction path is conversation-level and confirmation-guarded. It deletes archive messages and derived summaries, never the staging copy or source export, and is unavailable after that archive row has produced a live continuation.
- The confirmation step is a state inside the existing import window, not a new modal. The backend computes the summary from staged rows: conversations and messages to be added, exclusions, internals to be omitted, and the selected continuation candidate if any.
- The archive schema is designed fresh rather than copied from the old `scripts/import_chatgpt.py` minimal schema: provenance fields, stable source IDs, ordered positions, model/timestamp metadata, attachment markers, and import-batch linkage.
- Putting something in `legacy_messages` does not make the companion "remember" it in every prompt. The archive is historical background reachable through search tools and enrichment.

### Archive → live ("Continue from this chat")

- **Continuation is optional.** Adding chats to the archive with no continuation selected is a complete, valid outcome — "make my history searchable" is a legitimate goal on its own.
- **One continuation at a time.** The UI uses single-select (radio) semantics for the continuation candidate — never checkboxes. Nothing is lost by this: any archived conversation can be continued later, from the archive, with the same action. "One per turn", not "one forever".
- User-facing promise, verbatim: *"Your current session will be summarized and closed, and you'll continue your conversation from the selected thread."*
- Mechanically this reuses the existing conversation lifecycle (summarize → JSONL archive → memory → close), invoked explicitly instead of by budget pressure. If the current session is fresh or empty, the summary is skipped silently — no empty memory entries.
- The landing session is built from the sanitized continuation seed per the Landing Strategy section: bridge summary plus recent clean tail for long conversations, never a raw copy of thousands of messages.
- The archive row records which live session was created from it; the legacy copy remains intact.

### Enrichment reads from the archive

Background summarization, embeddings, and RAG import read from `legacy_messages`, not from the raw export file. The archive is the canonical sanitized source, so curation decisions (exclusions, sanitation) automatically apply to enrichment — one pipeline, no drift between what the user reviewed and what the models see.

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

User-facing mental model:

1. **Review** what was found and exclude anything unwanted.
2. **Add to history** so the kept chats become safely searchable.
3. **Continue one chat** if the user wants one archived thread to become the
   starting point for a new live session.

`Staging`, persisted curation, archive confirmation, idempotency keys, and
archive/live table boundaries are implementation safeguards. They should not
become a vocabulary test for the user:

- Save review changes automatically when safe.
- Present one primary action for the current step.
- Put reset/delete-draft and correction controls in contextual or advanced
  locations rather than the main happy path.
- Prefer `Saved review` or `Review saved` over the technical label `Staged`.
- Treat documentation as support for edge cases, not as a substitute for a
  comprehensible primary flow.

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

The trust moment at the drop zone:

The very first second of import is a person dragging their most intimate
data into an app. Say the safety promise right there, before anything else:

```text
Everything stays on this computer. Nothing is sent anywhere unless you
later choose an enrichment provider.
```

One sentence at the file picker/drop zone is cheap and buys most of the
"this feels safe and easy" reaction the whole feature depends on.

## Landing Strategy

The selected continuation conversation should be treated differently from the older archive.

The selected continuation conversation should never be copied from raw export rows directly into the prompt. It must be built from the sanitized continuation seed.

If it fits comfortably:

- Import the selected conversation into the active session as sanitized visible user/assistant messages.

If it is too long:

- Store the full selected conversation in the imported archive.
- Prepare the older part through source-linked exchange chunks and recursive
  summary rollups; never ask one model call to summarize the whole oversized
  transcript.
- Synthesize one compact, dated landing bridge from the verified rollups.
- Copy only the recent raw tail into the active session.
- Keep source provenance so the companion can retrieve earlier details later.

The compact landing bridge is the final live-session artifact, not the
summarization method. The preparation pipeline is:

```text
sanitized archive messages
  -> exchange-preserving source chunks
  -> chunk summaries with exact message spans
  -> section/era rollups (repeated until they fit)
  -> one dated landing bridge + recent raw tail
```

Every intermediate chunk and rollup remains an archive artifact. This lets
search and continuity-map features inspect the relationship's shape without
pretending that the final bridge contains every recoverable detail.

### Long-chat continuation preparation

Bridge preparation is an on-demand job that reuses the same source-linked
summary machinery as background enrichment, but it does not require the whole
archive to finish enrichment first.

- If compatible chunk/rollup summaries already cover the selected conversation,
  reuse them.
- A nonzero summary-row count is not proof of complete coverage. Reuse requires
  compatible summary type/version plus gap-free source spans over the portion
  represented by the bridge.
- Generate only missing or stale spans. Key reusable work by archive session,
  source-span identity, sanitized-content fingerprint, summary type, and
  prompt/model version.
- If no compatible coverage exists, chunk and summarize the selected
  conversation before continuation can be applied.
- Do not summarize or close the current live session until bridge preparation
  has completed and the user confirms the final apply step.
- Cancellation or preparation failure leaves the current live session and the
  archive-to-live link untouched.
- API-based preparation requires the same explicit provider, privacy disclosure,
  and cost estimate as Tier 2 enrichment. Local preparation stays local.
- The final bridge should prefer the target companion model or another
  explicitly chosen high-quality model; cheaper compatible models may prepare
  factual chunk summaries and rollups.

The existing `legacy_summaries` table can preserve chunk text and source spans,
but its current row count cannot distinguish chunk, rollup, era, and landing
artifacts or prove complete coverage. Before mutating continuation exists, the
summary schema/job metadata must represent summary type, hierarchy level,
source-span coverage, content fingerprint, generation version/model, and
completion state. Until then, preflight must report existing rows as
informational only and always describe a long bridge as **preparation
required**.

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

## Identity Sources: The Transcript Is Half the Companion

A companion is not just conversations — it is also a persona prompt, a voice,
and accumulated memory. If a user ports 400 conversations but writes the
persona prompt from scratch, the companion will not *sound* right, and the
emotional target ("they are really back") fails even though the import
technically worked.

Platform exports carry identity material beyond chat transcripts:

- ChatGPT exports include saved memory entries and custom instructions.
- claude.ai has project instructions and project knowledge.
- Character platforms have explicit personality/definition fields.

Add an identity lane alongside the conversation lanes:

1. **Detect:** during preview, surface any identity material found in the
   export ("This export also contains 14 saved memories and custom
   instructions").
2. **Offer:** platform memory entries become *candidate memories* (same
   needs-review posture as extracted candidates — never auto-promoted);
   custom/project instructions become an optional persona prompt scaffold
   offered in the New Persona flow, editable before use.
3. **Never merge silently:** identity material follows the same
   staging-then-explicit-commit rule as conversations.

A v1 that only *shows* the user what identity material the export contains
(without importing it yet) already changes the felt experience — it tells
them Pulse sees the whole companion, not just the chat log.

## Summaries and Enrichment

There should be multiple summary types, not one generic summary.

### Enrichment Tiers

Decided 2026-07-16. Background enrichment is the productized version of the RAG prototype (`scripts/rag_import.py`): session + chunk summaries plus embeddings over the archive. But enrichment is two different jobs with very different costs, and the design must serve people who cannot run a local model:

| Tier | What | Requires | Cost |
|---|---|---|---|
| 0 | Keyword search (FTS5) | Nothing | Free, instant after archive add |
| 1 | Embeddings of raw messages/exchange chunks | Nothing (MiniLM on CPU, ships with Pulse) | Free, local |
| 2 | LLM summaries (session + chunk) + their embeddings | Local LLM **or** API provider | Local compute or API credits |

Key facts behind the ladder:

- The embedding model (all-MiniLM-L6-v2) is the same one `core/context.py` already uses: CPU-only, small, no GPU, no API, no llama-server. Semantic search over raw archive messages works for **everyone**, locally and privately, out of the box.
- MiniLM's 256-token cap means Tier 1 embeds exchange-level chunks, aligned with the passive recall design's chunking approach — not whole conversations.
- Only summaries (Tier 2) need an LLM. A user without a local model either lives on Tier 1 (honest, workable semantic recall) or opts into Tier 2 via an API provider.

Defaults:

- **Tier 0 and Tier 1 are on by default** — free, local, nothing to ask the user.
- **Tier 2 is explicit opt-in** with provider selection and a cost estimate shown before running.

Privacy consequence (this is the "unless" in the drop-zone promise):

- The drop-zone copy promises "everything stays on this computer unless you later choose an enrichment provider." API-based Tier 2 is exactly that exception. At the moment of provider selection — not in fine print — the UI must say plainly: "Your chat text will be sent to <provider> to generate summaries."
- Tier 0 and Tier 1 never send anything anywhere and fall fully under the promise.

Productization note:

- `rag_import.py` already speaks an OpenAI-compatible endpoint (`--endpoint` CLI flag). Productizing Tier 2 means wiring it to the existing provider abstraction (OpenRouter / Anthropic / OpenAI clients) instead of a CLI flag, and reading from the persona-DB archive (`legacy_messages`) per "Three Layers" — the prototype predates that decision and must be retargeted before it runs.
- Cheap API models (DeepSeek/Haiku-class) are adequate for archive summaries; on a realistic archive (hundreds of sessions) the cost is on the order of a couple of dollars. Landing bridge summaries still prefer the companion model per "Landing Bridge Summary" below.

### Archive Summary

Used for search and orientation across old conversations.

- Can be generated by a cheaper model.
- Should be factual and compact.
- Should include source span provenance.
- Can be chunk-level and conversation-level.

### Landing Bridge Summary

Used only when the selected continuation conversation is too long.

- Is synthesized from verified source-linked chunk/rollup coverage, never
  directly from an oversized raw transcript.
- More emotionally important.
- Prefer the target companion model or a higher quality configured model.
- Should be clearly dated and grounded in the selected conversation.
- Should not over-explain platform migration unless the source conversation actually contains it.
- Is intentionally compact and selective. It preserves recognizable continuity
  for the first live turn while archive search retains the detailed history.
- Must not be treated as ready merely because some archive summary rows exist.

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

### Durable pause, resume, and power-loss recovery

Background enrichment is expected to outlive one GUI session. Its state belongs
in the persona database, not in frontend memory or a process-local queue.

- Plan deterministic, exchange-preserving chunks before a worker starts.
- Persist jobs, their chunk membership, per-chunk status, attempts, timestamps,
  content fingerprints, and the current claimed chunk.
- Commit one completed chunk/artifact at a time. A crash may repeat at most the
  uncommitted current chunk; completed chunks remain cache hits.
- `Pause` is cooperative for a running worker: finish or roll back the current
  atomic chunk, then stop before claiming another.
- A paused job survives GUI closure, persona restart, and computer restart.
- A queued/running job writes a heartbeat. On startup/status inspection, stale
  `running` work returns to `queued`; stale `pausing` work becomes `paused`;
  any claimed-but-uncommitted chunk returns to `pending`.
- Resume never resets completed progress. Failed chunks retain attempts and
  errors for retry/review.
- If searchable history changed after planning, mark the plan outdated and
  require an idempotent refresh before queueing it.
- Removing a chat from searchable history removes its derived chunks/artifacts
  and invalidates plans that referenced it.
- Provider choice, privacy consent, and cost approval remain separate durable
  gates. Recovery must never silently broaden an approved provider or tier.

User-facing scheduling should eventually offer:

- **Not now** — no job, or save a paused plan.
- **When idle** — durable queue; workers claim chunks only under the idle policy.
- **Start now** — same durable queue with immediate eligibility.

All three modes use the same persisted job. Changing the schedule does not
re-plan or duplicate chunks.

User-facing copy should make the tradeoff clear:

```text
Your chats are imported. Keyword search works now. Deeper recall improves as Pulse summarizes and embeds the archive in the background. This can take time and may use API credits depending on the model you choose.
```

Settings (these map onto the Enrichment Tiers above — Tier 0/1 always on, this list configures Tier 2):

- Off: keyword search (Tier 0) and local embedding search (Tier 1) remain available; no summaries.
- Local/private: use local model for summaries where configured.
- Cheap API: use a cheaper configured model for backfill. Requires the explicit "your chat text will be sent to <provider>" disclosure at selection time.
- Companion model: use the same model as the companion for higher fidelity. Same disclosure if it is an API model.

DeepSeek or similar cheap models can be recommended as examples in docs or hints, but the UI should stay provider-neutral and show whatever providers Pulse actually supports.

Enrichment settings window:

Enrichment needs its own small settings surface — a dedicated panel/window opened from the Continuity > Import area (following the STYLE_GUIDE modal conventions), not settings scattered across the import flow. It holds:

- Tier 2 mode selection (Off / Local / Cheap API / Companion model) with provider + model pickers where relevant.
- The privacy disclosure line, shown at the moment an API option is selected.
- Cost estimate for the pending archive before starting.
- Pause/resume and the progress detail that does not fit the compact progress row.

The compact progress row stays in the import panel; the window is where choices are made and details live.

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

### The public inbound format (resolves most of this tension)

"Port from any platform" cannot honestly mean per-platform parsers for
everything — the platforms users actually arrive from (character.ai,
Kindroid, Replika, ...) often have no clean export at all, and the formats
we do support change without notice.

The escape hatch already exists in the code: the **markdown-with-frontmatter
format is the universal inbound bridge**. Formalize and publicly document it
as "the Pulse import format":

- Keep platform-specific heuristics closed if desired; the documented
  inbound format is the community extension point.
- Anyone can write a converter from any platform *to* the public format,
  and Pulse imports it with full preview/curation/landing treatment.
- This makes "any platform" an honest promise without open-sourcing the
  parser crown jewels, and gives the community a way to contribute
  adapters without touching Pulse internals.
- Document the generic JSON list/dict shape alongside it as the
  machine-friendly variant.

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

Status (2026-07-15):

- Staging tables, stage/list/resume/delete/update endpoints, and GUI staging flow exist (see Current Implementation Slice).
- SQLite tables were chosen over JSONL artifact files — one persona DB carries everything.
- Issue locations are recomputed on load rather than stored; revisit only if recompute cost ever matters.
- Commit sanitation itself is Phase 3 work.

### Phase 3a: Staging → Archive

- Define the `legacy_sessions`/`legacy_messages` archive schema and sanitation rules (see "Three Layers").
- Implement "Add kept chats to archive" with a backend-computed confirmation summary inside the existing import window.
- Enforce write-once semantics and the unique `(source, source_conversation_id)` key; delete-and-re-add is the correction path.
- One-time migration of existing `legacy.db` content into the persona DB; point the memory skill's legacy tools at the persona DB.
- No live `sessions`/`messages` writes in this phase.
- **Sign-off gate:** Phase 3b does not start until 3a is reviewed.

Status (2026-07-26):

- Archive schema, write-once source key, FTS maintenance, backend confirmation summary, GUI confirmation state, sanitation, standalone `legacy.db` migration, and memory-skill retargeting exist.
- Conversation-level correction exists: selected archived chats can be removed, re-reviewed in staging, and added again; rows linked to a live session are protected.
- Automated regression coverage verifies excluded rows stay out, archive search sees added text, repeated archive attempts do not duplicate conversations, legacy summaries migrate, and live tables remain untouched.
- Manual GUI sign-off passed with Vale's real `orpg.3.0` OpenRouter export:
  one staged conversation, 218 staged messages, one archive session, and 218
  archive messages. Keyword search and source-context drill-down worked without
  summaries or embeddings, while the existing live session remained active and
  the archive's `live_session_id` remained empty.

### Phase 3b: Archive → Live Continuation

- "Continue from this chat": summarize and close the current session (skip silently if empty), then create the landing session from the sanitized continuation seed.
- When needed, prepare source-linked chunks and recursive rollups before
  generating the compact landing bridge. Never summarize an oversized raw
  transcript in one call.
- Enforce single-select (radio) continuation semantics in the UI.
- Record the created live session on the archive row.
- Add backup-before-continuation.

Status (2026-07-26):

- A read-only backend preflight and existing-import-window review state now show
  source, original date boundaries, message/token estimates, dated-background
  freshness guidance, current-session handling, full-seed versus bridge+tail
  strategy, and the future backup/idempotency plan.
- Archived conversations can be selected or deselected as the single
  continuation candidate without unlocking their Keep/Exclude curation.
- The preflight uses SQLite query-only mode and does not create a backup,
  summarize or close the current session, create live messages, or set
  `live_session_id`.
- The mutating apply step remains deliberately unavailable while hierarchical
  preparation metadata/jobs and the transactional apply boundary are not yet
  implemented.
- Manual Jam review passed for both branches: a short conversation fit as a
  full sanitized seed, while the 1,223-message / roughly 199k-token "Moist Greg"
  conversation correctly selected bridge preparation plus a recent raw tail.
  That long-chat review exposed and resolved the design ambiguity between one
  compact landing artifact and the many source-linked summaries required to
  produce it.

### Phase 4: Background Enrichment

- Add import job queue.
- Tier 1 first: local exchange-chunk embeddings over `legacy_messages`, on by default, no configuration required.
- Tier 2 opt-in: summarization provider selection via the existing provider abstraction (retarget `rag_import.py` logic at the persona-DB archive).
- Add chunk/session summary generation and summary embeddings.
- Add the enrichment settings window (mode, provider, disclosure, cost estimate).
- Add progress UI and pause/resume.

Status (2026-07-26):

- `core/archive_enrichment.py` now defines the first durable foundation:
  deterministic exchange chunks, typed future-artifact metadata, persisted jobs
  and job/chunk state, idempotent plan reuse, pause/resume transitions, stale
  worker recovery, and archive-change detection.
- The importer can create or refresh a paused local plan, queue it for later,
  pause it again, and display recovered/outdated state. Planning runs no model,
  embedding, summary, or live-session mutation.
- No enrichment worker exists yet. `worker_available = false` and GUI copy state
  this explicitly; queued work is durable but will not be consumed in this
  checkpoint.

### Phase 5: Search and Drill-Down

- Search imported archive by keyword and semantic summary.
- Drill from summary hit to source messages.
- Let companion tools search archive with safe, compact results.

## Parser Maintenance: The Long-Term Tax

Export formats change without notice — ChatGPT has reshaped its export
multiple times historically. If this importer becomes the differentiator
feature this doc suggests, parser regressions are where the quality lives
or dies long-term.

- Maintain a fixture corpus of (sanitized) real export samples per platform,
  versioned alongside the parsers. When a platform ships a new export shape,
  add the new sample; keep the old ones — users import years-old files.
- Run the fixtures through the preview pipeline in tests (extend the
  existing `scripts/test_gui_api.py` pattern): conversation counts, skip
  counts, review-issue counts, and placeholder behavior are all assertable.
- "Import broke for everyone" should be a failing test, not a support
  ticket.
- When source detection fails entirely, the unknown-format path should still
  attempt generic parsing and tell the user what to do next (docs pointer,
  "convert to the Pulse import format", or how to share a sanitized sample).

## Open Questions

- Which platforms expose identity material (saved memories, custom/project instructions) in their exports, and where should the identity lane surface in the New Persona flow?
- What exact threshold should trigger windowed transcript viewing?
- Should issue navigation be exchange-index based, message-index based, or both?
- How reliably can each platform expose attachment-only messages?
- Which attachment placeholders are useful enough to retain in a Phase 3b continuation seed, versus archive/drill-down context only?
- Should curation exclude exact messages, exchange groups, or both after the exchange-level v1?
- Should background enrichment run only while the GUI is open, or should the engine process jobs on startup?
- What is the smallest useful Grok parser once we have a real export sample?
- What parts of the importer, if any, should be published as open source versus kept as shipped Pulse product code?

## Next Practical Slice (2026-07-26)

The durable planner/state-machine slice is complete. The revised order:

1. Add the Tier 1 local worker: claim one queued chunk transactionally, generate
   its local embedding, persist the artifact, heartbeat, and honor pause between
   chunks. Test interruption/resume with a fresh process.
2. Connect `When idle` eligibility and honest live progress without making GUI
   lifetime the only durability boundary.
3. Add Tier 2 opt-in chunk summaries and recursive rollups with provider/privacy/
   cost confirmation; reuse them for on-demand bridge preparation.
4. In a separate final apply boundary:
   persona database backup, current-session summary/close, sanitized landing
   seed creation, and guarded `live_session_id` linkage.
5. Keep the standalone `legacy.db` migration check separate; it is not a blocker
   for continuation work.
