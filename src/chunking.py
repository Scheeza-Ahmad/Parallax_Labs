"""Lossless source slices with model-token windows and explicit overlap."""
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('Metadata must be a nonempty dictionary')
    for key, value in metadata.items():
        if not isinstance(key, str) or not key or type(value) not in (str, int, float, bool):
            raise ValueError(f'Invalid scalar metadata: {key!r}')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'Nonfinite metadata: {key}')


def chunk_document(record, tokenizer, chunk_size=220, overlap=32, max_seq_length=256):
    """Sizes are content tokens (special tokens excluded); offsets are Python characters.

    Windows use the model's fast tokenizer offsets, never decode token IDs, so
    case/Unicode/source text survive. Boundaries can split words or sentences.
    Each slice is retokenized to enforce the limit before it can be embedded.
    """
    if type(chunk_size) is not int or type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError('Require integer chunk_size > overlap >= 0')
    if chunk_size + tokenizer.num_special_tokens_to_add(pair=False) > max_seq_length:
        raise ValueError('chunk_size plus special tokens exceeds model input limit')
    if not isinstance(record, dict):
        raise ValueError('Document must be a dictionary')
    for key in ('doc_id', 'source', 'source_url'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'Missing source field: {key}')
    text = record.get('text')
    if not isinstance(text, str):
        raise ValueError('Document text must be a string')
    metadata = {k: v for k, v in record.items() if k != 'text'}
    validate_metadata(metadata)
    reserved = {'chunk_index', 'char_start', 'char_end', 'token_start', 'token_end', 'token_count'}
    if reserved & metadata.keys():
        raise ValueError('Source metadata uses reserved chunk fields')
    if not text.strip():
        return []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    offsets = encoded['offset_mapping']
    if not offsets:
        raise ValueError('Nonempty text produced no model tokens')
    chunks, start = [], 0
    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))
        left = 0 if start == 0 else offsets[start][0]
        while end > start:
            right = len(text) if end == len(offsets) else offsets[end][0]
            piece = text[left:right]
            count = len(tokenizer(piece, add_special_tokens=False, truncation=False)['input_ids'])
            if piece.strip() and count <= chunk_size:
                break
            end -= 1
        if end <= start or (end < len(offsets) and end - start <= overlap):
            raise ValueError('Cannot make progress within token budget; reduce overlap')
        meta = dict(metadata, chunk_index=len(chunks), char_start=left, char_end=right,
                    token_start=start, token_end=end, token_count=count)
        identity = json.dumps([meta, piece, chunk_size, overlap, 'token-window-v1'],
                              ensure_ascii=False, sort_keys=True)
        chunks.append(Chunk(hashlib.sha256(identity.encode('utf-8')).hexdigest(), piece, meta))
        if end == len(offsets):
            break
        start = end - overlap
    return chunks
