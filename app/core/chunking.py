"""Text chunking utilities for Precedent V1 (FR-1.x)."""

from __future__ import annotations

from typing import List, Optional


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_size: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk (smaller trailing chunks are merged)

    Returns:
        List of text chunks

    Raises:
        ValueError: If text is empty, chunk_size <= 0, or overlap >= chunk_size
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty or whitespace only")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")

    text = text.strip()
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]

        # If this is the last chunk and it's too small, try to merge with previous
        if end == text_len and len(chunk) < min_chunk_size and chunks:
            # Merge with previous chunk
            chunks[-1] = chunks[-1] + chunk
        else:
            chunks.append(chunk)

        if end == text_len:
            break

        start = end - overlap

    return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split text into chunks by sentence boundaries, respecting max_chunk_size.
    Attempts to keep sentences intact within chunks.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum chunk size in characters
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty or whitespace only")

    # Simple sentence splitting (can be improved with nltk/spacy)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    if not sentences:
        return [text]

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_size = len(sentence)

        # If single sentence exceeds max, fall back to character chunking
        if sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            # Chunk the long sentence
            sub_chunks = chunk_text(sentence, chunk_size=max_chunk_size, overlap=max_chunk_size // 10)
            chunks.extend(sub_chunks)
            continue

        # Check if adding this sentence would exceed max size
        if current_size + sentence_size + (1 if current_chunk else 0) > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(chunks) > 0:
                # Get last N sentences from previous chunk for overlap
                prev_sentences = chunks[-1].split(". ")
                overlap_text = ". ".join(prev_sentences[-overlap_sentences:])
                current_chunk = [overlap_text, sentence]
                current_size = len(overlap_text) + sentence_size + 2
            else:
                current_chunk = [sentence]
                current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + (1 if current_chunk[:-1] else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """
    Rough estimate of token count for a text.
    Default assumes ~4 chars per token (typical for English).
    """
    return max(1, int(len(text) / chars_per_token))