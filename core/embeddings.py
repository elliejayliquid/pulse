import numpy as np
import re

def blob_to_vec(blob) -> np.ndarray | None:
    """Convert an embedding blob (bytes/list/ndarray) to a numpy array."""
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        if not blob:
            return None
        try:
            return np.frombuffer(blob, dtype=np.float32).copy()
        except Exception:
            return None
    if isinstance(blob, list):
        if not blob:
            return None
        try:
            return np.array(blob, dtype=np.float32)
        except Exception:
            return None
    if isinstance(blob, np.ndarray):
        return blob
    return None

def embedding_to_blob(vec) -> bytes | None:
    """Convert a vector (list/ndarray) to SQLite BLOB bytes."""
    if vec is None or (hasattr(vec, '__len__') and len(vec) == 0):
        return None
    try:
        return np.array(vec, dtype=np.float32).tobytes()
    except Exception:
        return None

def chunk_text(text: str, max_chars: int = 600, max_chunks: int = 8) -> list[str]:
    """Split text into chunks suitable for the embedding model's cap.
    
    1. Splits on blank lines into paragraphs.
    2. Paragraphs longer than max_chars are split into sentence groups.
    3. Blocks of text with no punctuation breaks are hard-split at max_chars.
    4. Drops chunks under 20 characters.
    5. Caps at max_chunks, keeping the first and last chunks if capped.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks = []

    for p in paragraphs:
        if len(p) <= max_chars:
            raw_chunks.append(p)
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', p) if s.strip()]
            current_chunk = []
            current_len = 0
            
            for sentence in sentences:
                sentence_len = len(sentence)
                if sentence_len > max_chars:
                    if current_chunk:
                        raw_chunks.append(" ".join(current_chunk).strip())
                        current_chunk = []
                        current_len = 0
                    
                    # Hard-split long sentence/wall-of-text
                    start = 0
                    while start < sentence_len:
                        raw_chunks.append(sentence[start:start+max_chars])
                        start += max_chars
                else:
                    space_len = 1 if current_chunk else 0
                    if current_len + space_len + sentence_len > max_chars:
                        raw_chunks.append(" ".join(current_chunk).strip())
                        current_chunk = [sentence]
                        current_len = sentence_len
                    else:
                        current_chunk.append(sentence)
                        current_len += space_len + sentence_len
            
            if current_chunk:
                raw_chunks.append(" ".join(current_chunk).strip())

    # Drop chunks under 20 chars
    filtered_chunks = [c for c in raw_chunks if len(c) >= 20]

    # Cap at max_chunks, keeping the first and last chunks if capping
    if len(filtered_chunks) > max_chunks:
        half = max_chunks // 2
        filtered_chunks = filtered_chunks[:half] + filtered_chunks[-(max_chunks - half):]

    return filtered_chunks

def max_similarity(query_vecs: list[np.ndarray], mem_vecs: list[np.ndarray]) -> float:
    """Calculate max cosine similarity between any query chunk and any memory vector."""
    if not query_vecs or not mem_vecs:
        return 0.0

    best_sim = 0.0
    normed_mems = []
    
    for mv in mem_vecs:
        if mv is None or len(mv) == 0:
            continue
        nm = np.linalg.norm(mv)
        if nm > 0:
            normed_mems.append((mv, nm))

    if not normed_mems:
        return 0.0

    for q_vec in query_vecs:
        if q_vec is None or len(q_vec) == 0:
            continue
        nq = np.linalg.norm(q_vec)
        if nq == 0:
            continue
        for mv, nm in normed_mems:
            sim = float(np.dot(q_vec, mv) / (nq * nm))
            if sim > best_sim:
                best_sim = sim

    return best_sim
