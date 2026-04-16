#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Phase 2: Semantic RAG for legacy conversations
# Reads sessions from legacy.db, generates summaries + embeddings via local llama-server,
# splits long sessions into chunks, stores everything in legacy_summaries.
# Make sure you have a working llama-server running at http://127.0.0.1:8011/v1
#
# Usage:
#   python rag_import.py                           # dry run (print what would be done)
#   python rag_import.py --execute                 # actually write to DB
#   python rag_import.py --execute --reprocess     # re-summarize sessions that already have summaries

import argparse
import json
import logging
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path
from datetime import datetime

# Add pulse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- Logging Setup ---
_root = logging.getLogger()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
_handler.flush = sys.stdout.flush
_root.addHandler(_handler)
# CHANGED: Set to DEBUG for ultimate visibility while you troubleshoot!
_root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_ENDPOINT = os.environ.get(
    'LLAMA_ENDPOINT', 'http://127.0.0.1:8011/v1'
)
LLAMA_MODEL = os.environ.get('LLAMA_MODEL', 'default')
LLAMA_SUMMARIZE_TEMPERATURE = 0.4
LLAMA_SUMMARIZE_MAX_TOKENS = 512  # enough for reasoning models (thinking + summary)
LLAMA_MAX_RETRIES = 5
LLAMA_RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff
LLAMA_COOLDOWN_DELAY = 1.5         # seconds to pause after a successful call

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # must match core/context.py

CHUNK_SIZE = 10  # messages per sub-chunk
CHUNK_OVERLAP = 2  # overlapping messages between chunks
LONG_SESSION_THRESHOLD = 15  # sessions above this get sub-chunked

LEGACY_DB = Path(os.environ.get(
    'LEGACY_DB',
    '/path/to/your/legacy.db'
))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_embedding_model():
    # LOG: Function Entry
    logger.debug("-> ENTER: load_embedding_model()")
    logger.info(f'Loading embedding model ({EMBEDDING_MODEL_NAME}) on CPU...')
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        logger.info('Embedding model loaded successfully.')
        # LOG: Function Exit (Success)
        logger.debug("<- EXIT: load_embedding_model() - Success")
        return model
    except Exception as e:
        # LOG: Function Exit (Failure)
        logger.error(f'!!! FAILED to load embedding model: {e}')
        logger.debug("<- EXIT: load_embedding_model() - Exception Caught")
        return None


def embed_text(model, text: str) -> list[float] | None:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: embed_text() - Text length: {len(text)} chars")
    if model is None:
        logger.warning("Embed text aborted: model is None.")
        logger.debug("<- EXIT: embed_text() - Model is None")
        return None

    logger.debug("Generating embeddings via model.encode()...")
    vec = model.encode(text, normalize_embeddings=True)

    # LOG: Function Exit
    logger.debug("<- EXIT: embed_text() - Embedding generated successfully")
    return vec.tolist()


def embed_to_blob(vec: list[float] | None) -> bytes | None:
    # LOG: Function Entry
    logger.debug("-> ENTER: embed_to_blob()")
    if vec is None:
        logger.debug("<- EXIT: embed_to_blob() - Vector is None, returning None")
        return None

    # LOG: Conversion step
    logger.debug(f"Packing vector of length {len(vec)} into bytes...")
    blob = struct.pack(f'{len(vec)}f', *vec)

    # LOG: Function Exit
    logger.debug(f"<- EXIT: embed_to_blob() - Packed into {len(blob)} bytes")
    return blob


def blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    # LOG: Function Entry
    logger.debug("-> ENTER: blob_to_vec()")
    if not blob:
        logger.debug("<- EXIT: blob_to_vec() - Blob is None/Empty")
        return None

    logger.debug("Unpacking blob into numpy array...")
    vec = np.frombuffer(blob, dtype=np.float32).copy()

    # LOG: Function Exit
    logger.debug(f"<- EXIT: blob_to_vec() - Array shape: {vec.shape}")
    return vec


def chunk_messages(messages: list[dict], size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[list[dict]]:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: chunk_messages() - Messages: {len(messages)}, Size: {size}, Overlap: {overlap}")

    if len(messages) <= LONG_SESSION_THRESHOLD:
        logger.debug("<- EXIT: chunk_messages() - Below threshold, returning empty chunk list")
        return []  # no sub-chunks needed

    chunks = []
    start = 0
    while start < len(messages):
        end = min(start + size, len(messages))
        chunks.append(messages[start:end])
        logger.debug(f"Created chunk from index {start} to {end}")

        # THE FIX: If we've reached the end of the messages, break the loop
        # so the overlap calculation doesn't drag us backwards forever!
        if end == len(messages):
            logger.debug("Reached end of messages. Breaking out of chunk loop!")
            break

        start = end - overlap

    # LOG: Function Exit
    logger.debug(f"<- EXIT: chunk_messages() - Returned {len(chunks)} chunks")
    return chunks


# ── LLM summarization ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    'You are an expert conversational summarizer.\n'
    'Given a dialogue, write 3 original sentences that capture the essence.\n'
    'Do NOT repeat phrases from the original. Do NOT echo the messages back.\n'
    'Do NOT output [USER] or [ASSISTANT] tags in your summary.\n'
    'Write entirely NEW sentences.\n'
    '\n'
    'Example:\n'
    'Input: [USER] Hi! [ASSISTANT] Hey there!\n'
    'Summary: A casual greeting between two people reconnecting after time apart.\n'
    '\n'
    'Now summarize this conversation:'
)


def build_messages_text(messages: list[dict]) -> str:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: build_messages_text() - Processing {len(messages)} messages")
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '').strip()
        if content:
            lines.append(f'[{role.upper()}] {content}')
        else:
            logger.debug(f"Skipping empty message at index {i}")

    result = '\n'.join(lines)
    # LOG: Function Exit
    logger.debug(f"<- EXIT: build_messages_text() - Result length: {len(result)} chars")
    return result


def summarize_prompt(messages: list[dict]) -> str:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: summarize_prompt() - Determining sentence count for {len(messages)} messages")

    count = len(messages)
    if count <= 4:
        sentences = '2'
    elif count <= 20:
        sentences = '3'
    else:
        sentences = '4'

    logger.debug(f"Selected {sentences} sentences for summary.")
    messages_text = build_messages_text(messages)

    prompt = (
        f'Summarize the following conversation in {sentences} sentences.\n'
        f'Capture the main topics discussed and the overall tone.\n'
        f'\n--- Conversation ---\n{messages_text}\n'
        f'\n--- Summary ---\n'
    )
    # LOG: Function Exit
    logger.debug("<- EXIT: summarize_prompt() - Prompt assembled")
    return prompt


def estimate_tokens(text: str) -> int:
    # LOG: Function Entry
    logger.debug("-> ENTER: estimate_tokens()")

    est = int(len(text.split()) * 1.3)

    # LOG: Function Exit
    logger.debug(f"<- EXIT: estimate_tokens() - Estimated {est} tokens")
    return est


MAX_CONTEXT_TOKENS = int(os.environ.get('LLAMA_MAX_CONTEXT', 16384))
SAFE_CONTEXT = int(MAX_CONTEXT_TOKENS * 0.75)  # leave room for prompt overhead


def summarize_messages(messages: list[dict], client: OpenAI) -> str:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: summarize_messages() - Attempting to summarize {len(messages)} messages")

    if not messages:
        logger.warning("summarize_messages received empty message list.")
        logger.debug("<- EXIT: summarize_messages() - Empty messages")
        return ''

    prompt = summarize_prompt(messages)

    # Truncate messages if they exceed safe context window
    token_count = estimate_tokens(prompt)
    logger.debug(f"Estimated token count: {token_count} (Safe Context Limit: {SAFE_CONTEXT})")

    if token_count > SAFE_CONTEXT:
        logger.warning(f"Token count ({token_count}) exceeds SAFE_CONTEXT ({SAFE_CONTEXT}). Truncating...")
        # Keep the last messages that fit — they're most recent and relevant
        target_tokens = SAFE_CONTEXT - estimate_tokens(
            SYSTEM_PROMPT + summarize_prompt([])
        )
        truncated = []
        for msg in reversed(messages):
            text = f'[{msg["role"].upper()}] {msg["content"]}'
            tok = estimate_tokens(text)
            if target_tokens - tok >= 0:
                target_tokens -= tok
                truncated.insert(0, msg)
            else:
                logger.debug(f"Hit token limit. Discarding remaining older messages.")
                break

        if len(truncated) < len(messages):
            logger.warning(f'Truncated {len(messages)} msgs down to {len(truncated)} to fit context')
        messages = truncated
        prompt = summarize_prompt(messages)

    last_exc = None
    raw = ''

    # Retry loop for LLM calls
    for attempt in range(LLAMA_MAX_RETRIES):
        logger.debug(f"LLM API Call Attempt {attempt + 1}/{LLAMA_MAX_RETRIES}...")
        try:
            logger.info("Calling LLM completion endpoint...")
            response = client.chat.completions.create(
                model=LLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=LLAMA_SUMMARIZE_TEMPERATURE,
                max_tokens=LLAMA_SUMMARIZE_MAX_TOKENS,
                timeout=60
            )
            msg = response.choices[0].message
            # Reasoning models (Ministral, Qwen3, etc.) may put thinking in reasoning_content
            raw = msg.content or getattr(msg, 'reasoning_content', None) or ''
            logger.debug(f"LLM Call Success. Received {len(raw)} chars of raw output.")

            # --- NEW BREATHER ADDED HERE ---
            logger.info(f"Cooling down for {LLAMA_COOLDOWN_DELAY}s...")
            time.sleep(LLAMA_COOLDOWN_DELAY)

            break

        except Exception as e:
            last_exc = e
            logger.error(f'!!! LLM API Exception: {e}')
            if attempt < LLAMA_MAX_RETRIES - 1:
                delay = LLAMA_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f'LLM call failed, backing off... Retrying in {delay:.0f}s')
                time.sleep(delay)
            else:
                logger.error(f'!!! FATAL: LLM call failed completely after {LLAMA_MAX_RETRIES} attempts.')
                logger.debug("<- EXIT: summarize_messages() - API Failure")
                return ''

    if not raw:
        logger.warning("LLM returned empty output.")
        logger.debug("<- EXIT: summarize_messages() - Empty Output")
        return ''

    logger.debug("Cleaning LLM output (stripping think tags)...")
    # Strip think tags (covers, Thinking Process:, etc.)
    try:
        from core.llm import strip_think_tags
        text = strip_think_tags(raw)
        logger.debug("Successfully applied strip_think_tags.")
    except ImportError as e:
        logger.error(f"Could not import strip_think_tags from core.llm: {e}")
        text = raw

    # Robust fallback: if stripped text is empty, split on "---" divider
    if not text or len(text) < 20:
        logger.debug("Output looks too short/empty after stripping tags. Applying fallback '---' split.")
        parts = raw.split('---')
        text = parts[-1].strip() if len(parts) > 1 else raw[-400:].strip()

    # Strip trailing self-critique lines the model appended
    logger.debug("Running regex to clean up critique lines...")
    import re
    text = re.sub(r'\n\s*\*\s*\*[^:]+\*:[^\n]*\n?', '', text).strip()
    text = re.sub(r'\n[*\d\s]+(?:Critique|Notes?|Feedback|Score)[^:\n]*\n?$', '', text,
                  flags=re.IGNORECASE).strip()

    # Echo-back detection: if output starts with role tags, grab last 300 chars
    if text.strip().startswith('[USER]') or text.strip().startswith('[ASSISTANT]'):
        logger.debug("Echo-back detected in LLM output! Truncating to last 300 chars.")
        text = raw[-300:].strip()

    logger.info("LLM response successfully processed and cleaned.")

    # LOG: Function Exit
    logger.debug("<- EXIT: summarize_messages() - Success")
    return text


# ── Database ───────────────────────────────────────────────────────────────────

def open_legacy_db(db_path: Path) -> sqlite3.Connection:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: open_legacy_db() - Path: {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        logger.debug("Database connection established (WAL mode).")

        # LOG: Function Exit
        logger.debug("<- EXIT: open_legacy_db() - Success")
        return conn
    except Exception as e:
        logger.error(f"!!! Failed to open database at {db_path}: {e}")
        logger.debug("<- EXIT: open_legacy_db() - Failure")
        raise


def ensure_summaries_table(conn: sqlite3.Connection):
    # LOG: Function Entry
    logger.debug("-> ENTER: ensure_summaries_table()")

    try:
        logger.debug("Executing table creation scripts...")
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS legacy_summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                summary_text    TEXT NOT NULL,
                summary_embedding BLOB,
                chunk_index     INTEGER DEFAULT 0,
                first_msg_id    INTEGER,
                last_msg_id     INTEGER,
                is_topic_boundary INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_ls_session ON legacy_summaries(session_id);
            CREATE INDEX IF NOT EXISTS idx_ls_chunk ON legacy_summaries(session_id, chunk_index);
        ''')
        conn.commit()
        logger.debug("Tables and indices verified/created.")

        # LOG: Function Exit
        logger.debug("<- EXIT: ensure_summaries_table() - Success")
    except Exception as e:
        logger.error(f"!!! Database error while ensuring tables: {e}")
        logger.debug("<- EXIT: ensure_summaries_table() - Failure")
        raise


def already_has_summary(conn: sqlite3.Connection, session_id: str,
                        chunk_index: int) -> bool:
    # LOG: Function Entry
    logger.debug(f"-> ENTER: already_has_summary() - Session: {session_id}, Chunk: {chunk_index}")

    try:
        row = conn.execute(
            'SELECT id FROM legacy_summaries WHERE session_id=? AND chunk_index=?',
            (session_id, chunk_index)
        ).fetchone()

        exists = row is not None
        logger.debug(f"Summary existence check for {session_id} (Chunk {chunk_index}): {exists}")

        # LOG: Function Exit
        logger.debug("<- EXIT: already_has_summary()")
        return exists
    except Exception as e:
        logger.error(f"!!! Database query failed in already_has_summary: {e}")
        logger.debug("<- EXIT: already_has_summary() - Failure")
        raise


def upsert_summary(conn: sqlite3.Connection, session_id: str,
                   summary_text: str, embedding_blob: bytes | None,
                   chunk_index: int, first_msg_id: int | None,
                   last_msg_id: int | None, is_topic_boundary: int = 0):
    # LOG: Function Entry
    logger.debug(f"-> ENTER: upsert_summary() - Session: {session_id}, Chunk: {chunk_index}")

    try:
        logger.debug(f"Checking for existing record to UPDATE vs INSERT...")
        existing = conn.execute(
            'SELECT id FROM legacy_summaries WHERE session_id=? AND chunk_index=?',
            (session_id, chunk_index)
        ).fetchone()

        if existing:
            logger.debug(f"Record found (ID {existing['id']}). Executing UPDATE.")
            conn.execute(
                '''UPDATE legacy_summaries
                   SET summary_text=?, summary_embedding=?, first_msg_id=?, last_msg_id=?, is_topic_boundary=?
                   WHERE id=?''',
                (summary_text, embedding_blob, first_msg_id, last_msg_id, is_topic_boundary, existing['id'])
            )
        else:
            logger.debug("No existing record. Executing INSERT.")
            conn.execute(
                '''INSERT INTO legacy_summaries
                   (session_id, summary_text, summary_embedding, chunk_index, first_msg_id, last_msg_id, is_topic_boundary)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (session_id, summary_text, embedding_blob, chunk_index,
                 first_msg_id, last_msg_id, is_topic_boundary)
            )

        # LOG: Function Exit
        logger.debug("<- EXIT: upsert_summary() - Query staged (commit happens upstream)")
    except Exception as e:
        logger.error(f"!!! Failed to upsert summary for session {session_id}: {e}")
        logger.debug("<- EXIT: upsert_summary() - Failure")
        raise


# ── Main ───────────────────────────────────────────────────────────────────────

def run(dry_run: bool = True, reprocess: bool = False,
        batch_pause: int = 20,
        endpoint: str | None = None):
    # LOG: Function Entry
    logger.debug(f"-> ENTER: run() - DryRun: {dry_run}, Reprocess: {reprocess}, BatchPause: {batch_pause}")

    # Resolve endpoint at call time (not module load) so CLI env overrides work
    llm_endpoint = endpoint or os.environ.get('LLAMA_ENDPOINT', 'http://127.0.0.1:8011/v1')

    logger.info(f'Opening legacy.db: {LEGACY_DB}')
    if not LEGACY_DB.exists():
        logger.error(f'!!! Database not found: {LEGACY_DB}')
        sys.exit(1)

    logger.debug("Calling open_legacy_db()...")
    conn = open_legacy_db(LEGACY_DB)

    logger.debug("Calling ensure_summaries_table()...")
    ensure_summaries_table(conn)

    # Init embedding model
    logger.debug("Calling load_embedding_model()...")
    embed_model = load_embedding_model()

    # Init LLM client
    logger.debug("Initializing OpenAI Client...")
    llm_client = OpenAI(
        base_url=llm_endpoint,
        api_key='not-needed',
        timeout=60.0  # or even 30 if you want it aggressive
    )
    logger.info(f'LLM endpoint config: {llm_endpoint}')
    logger.info(f'Embedding model config: {EMBEDDING_MODEL_NAME if embed_model else None}')

    # Count sessions
    logger.debug("Querying database for all legacy_sessions...")
    sessions = conn.execute(
        'SELECT id, title, message_count FROM legacy_sessions ORDER BY created_at'
    ).fetchall()

    total = len(sessions)
    logger.info(f'Found {total} sessions to process.')

    processed = 0
    skipped = 0
    errors = 0

    # Main Processing Loop
    for row_idx, row in enumerate(sessions):
        session_id = row['id']
        title = row['title'] or session_id

        logger.debug(f"--- Processing Session {row_idx + 1}/{total} ---")
        logger.info(f'Started processing session: {title} ({session_id})')

        # Get all messages for this session
        logger.debug(f"Fetching messages for session {session_id}...")
        messages = conn.execute(
            'SELECT id, role, content FROM legacy_messages WHERE session_id=? ORDER BY id',
            (session_id,)
        ).fetchall()
        messages = [dict(r) for r in messages]
        logger.debug(f"Retrieved {len(messages)} messages for session {session_id}.")

        if not messages:
            skipped += 1
            logger.debug(f'Skipping empty session (no messages): {session_id}')
            continue

        # ── Session-level summary (chunk_index = 0) ─────────────────────────
        logger.info(f'Generating session-level summary (chunk 0) for {session_id}')

        needs_processing = reprocess or not already_has_summary(conn, session_id, 0)
        logger.debug(f"Needs processing check (chunk 0): {needs_processing}")

        if needs_processing:
            if dry_run:
                logger.info(f'[DRY RUN] Would summarize session {session_id}: {title}')
            else:
                logger.info(f'Summarizing full session: {session_id} ({len(messages)} msgs)')

                logger.debug(f"Calling summarize_messages() for full session...")
                summary_text = summarize_messages(messages, llm_client)

                if summary_text:
                    logger.debug(f"Generating embeddings for summary...")
                    vec = embed_text(embed_model, summary_text)
                    blob = embed_to_blob(vec)

                    logger.debug(f"Upserting session summary to database...")
                    upsert_summary(conn, session_id, summary_text, blob,
                                   chunk_index=0,
                                   first_msg_id=messages[0]['id'],
                                   last_msg_id=messages[-1]['id'],
                                   is_topic_boundary=0)
                    conn.commit()
                    logger.debug(f"Database commit successful for session {session_id}.")
                    processed += 1
                    logger.info(f'  → Success: \"{summary_text[:80]}...\"')
                else:
                    errors += 1
                    logger.warning(f'  → !!! Summarization failed to return text for {session_id}')
        else:
            skipped += 1
            logger.debug(f"Skipped chunk 0 for {session_id} (already exists and not reprocessing)")

        # ── Sub-chunks for long sessions ───────────────────────────────────
        logger.info(f'Evaluating sub-chunks for long sessions...')

        if len(messages) > LONG_SESSION_THRESHOLD:
            logger.debug(f"Session exceeds threshold ({len(messages)} > {LONG_SESSION_THRESHOLD}). Chunking...")
            chunks = chunk_messages(messages, CHUNK_SIZE, CHUNK_OVERLAP)
            logger.info(f'  Split {session_id} into {len(chunks)} sub-chunks')

            for ci, chunk in enumerate(chunks, start=1):
                logger.debug(f"Processing chunk {ci}/{len(chunks)} for session {session_id}...")

                chunk_needs_processing = reprocess or not already_has_summary(conn, session_id, ci)
                logger.debug(f"Chunk {ci} needs processing: {chunk_needs_processing}")

                if chunk_needs_processing:
                    if dry_run:
                        logger.debug(f"[DRY RUN] Would process chunk {ci}")
                        continue

                    logger.debug(f"Calling summarize_messages() for chunk {ci}...")
                    chunk_text = summarize_messages(chunk, llm_client)

                    if chunk_text:
                        logger.info(f'Sub-chunk {ci} successfully summarized.')
                        vec = embed_text(embed_model, chunk_text)
                        blob = embed_to_blob(vec)

                        logger.debug(f"Upserting chunk {ci} summary to database...")
                        upsert_summary(conn, session_id, chunk_text, blob,
                                       chunk_index=ci,
                                       first_msg_id=chunk[0]['id'],
                                       last_msg_id=chunk[-1]['id'],
                                       is_topic_boundary=0)
                        conn.commit()
                        logger.debug(f"Database commit successful for chunk {ci}.")
                        processed += 1
                    else:
                        logger.warning(f"!!! Failed to summarize chunk {ci} for session {session_id}")
                        errors += 1

                # Pause every batch_pause chunks to avoid hammering the LLM
                if not dry_run and (processed % batch_pause == 0) and processed > 0:
                    logger.info(f'  ... Reached batch pause limit ({batch_pause}). Sleeping for 1s to cool down API.')
                    time.sleep(1)
        else:
            logger.debug(f"Session {session_id} is under threshold ({len(messages)}). No sub-chunks needed.")

    # Cleanup and Exit
    logger.debug("Finished main loop. Closing database connection...")
    conn.close()

    logger.info(
        f'Done. processed={processed}, skipped={skipped}, errors={errors}'
    )
    # LOG: Function Exit
    logger.debug("<- EXIT: run() - Script complete")


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser(description='Phase 2 RAG import for legacy.db')
    parser.add_argument('--execute', action='store_true',
                        help='Actually write to DB (default is dry-run)')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-summarize sessions that already have summaries')
    parser.add_argument('--batch-pause', type=int, default=20,
                        help='Pause every N LLM calls to avoid overload (default 20)')
    parser.add_argument('--db', type=str, default=None,
                        help=f'Override LEGACY_DB path')
    parser.add_argument('--endpoint', type=str, default=None,
                        help='Override LLM endpoint')
    parser.add_argument('--model', type=str, default=None,
                        help='Override LLM model name')
    args = parser.parse_args()

    if args.db:
        os.environ['LEGACY_DB'] = args.db
    if args.endpoint:
        os.environ['LLAMA_ENDPOINT'] = args.endpoint
    if args.model:
        os.environ['LLAMA_MODEL'] = args.model

    dry_run = not args.execute
    if dry_run:
        logger.info('DRY RUN — use --execute to actually write to DB')

    run(dry_run=dry_run, reprocess=args.reprocess, batch_pause=args.batch_pause,
        endpoint=args.endpoint)