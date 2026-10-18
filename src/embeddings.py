"""Pinned local sentence-transformer encoder, shared by indexing and queries."""
import os
from pathlib import Path
from time import perf_counter

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


class Encoder:
    def __init__(self, model=MODEL, revision=REVISION, device='cpu', batch_size=32,
                 cache='.cache/huggingface/hub', offline=False, threads=4):
        if batch_size < 1 or threads < 1:
            raise ValueError('batch_size and threads must be positive')
        os.environ.setdefault('HF_HOME', str(Path('.cache/huggingface').resolve()))
        import torch
        from sentence_transformers import SentenceTransformer
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; explicitly select --device cpu for CPU development')
        torch.set_num_threads(threads)
        started = perf_counter()
        self.model = SentenceTransformer(model, revision=revision, device=device,
                                         cache_folder=cache, local_files_only=offline)
        self.load_seconds = perf_counter() - started
        self.tokenizer = self.model.tokenizer
        if not self.tokenizer.is_fast:
            raise ValueError('A fast tokenizer with source offsets is required')
        self.batch_size = batch_size
        self.max_seq_length = self.model.max_seq_length
        self.identity = dict(model=model, revision=revision,
                             dimension=self.model.get_embedding_dimension(),
                             normalized=True, max_seq_length=self.max_seq_length)
        self.details = dict(self.identity, device=str(self.model.device), batch_size=batch_size,
                            threads=threads, model_load_seconds=self.load_seconds)

    def encode(self, texts):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Encoding requires nonempty texts')
        for text in texts:
            if len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_seq_length:
                raise ValueError('Input exceeds model limit; chunk document or shorten query')
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
