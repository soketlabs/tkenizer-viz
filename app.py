"""
Tokenizer Visualization Server
-------------------------------
A FastAPI backend that tokenizes text using multiple HuggingFace tokenizers
and returns token-level information for visualization.

To add a new tokenizer, simply append its HuggingFace model identifier
to the TOKENIZERS list below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer

# ──────────────────────────────────────────────
#  ADD / REMOVE TOKENIZERS HERE
# ──────────────────────────────────────────────
TOKENIZERS: list[str] = [
    "soketlabs/Eka_tokenizer",
    "sarvamai/sarvam-1",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-2b",
    "mistralai/Mistral-7B-v0.1",
    "openai-community/gpt2",
    "Qwen/Qwen2.5-7B",
    "deepseek-ai/DeepSeek-V3",
]
# ──────────────────────────────────────────────

app = FastAPI(title="Tokenizer Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded tokenizers
_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(name: str) -> Any:
    """Load a tokenizer (cached)."""
    if name not in _tokenizer_cache:
        print(f"  Loading tokenizer: {name} ...")
        _tokenizer_cache[name] = AutoTokenizer.from_pretrained(name)
        print(f"  ✓ {name} loaded")
    return _tokenizer_cache[name]


def preload_tokenizers() -> None:
    """Pre-load all tokenizers at startup so first request is fast."""
    print("Pre-loading tokenizers …")
    for name in TOKENIZERS:
        try:
            _get_tokenizer(name)
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    print("All tokenizers ready.\n")


# --------------- Models ---------------

class TokenizeRequest(BaseModel):
    text: str


class TokenInfo(BaseModel):
    token_id: int
    token_str: str
    start: int | None  # char offset in original text
    end: int | None


class TokenizerResult(BaseModel):
    model_name: str
    num_tokens: int
    num_chars: int
    tokens: list[TokenInfo]


class TokenizeResponse(BaseModel):
    results: list[TokenizerResult]


# --------------- Helpers ---------------

def _token_spans(tokenizer, text: str) -> list[TokenInfo]:
    """
    Return a list of TokenInfo with character-level offsets.
    Uses the fast tokenizer's offset mapping when available,
    otherwise falls back to decoding individual token ids.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    ids: list[int] = encoding["input_ids"]

    # Fast tokenizers provide offset_mapping
    if "offset_mapping" in encoding and encoding["offset_mapping"] is not None:
        spans = encoding["offset_mapping"]
        result = []
        for tid, (s, e) in zip(ids, spans):
            token_str = text[s:e] if s is not None and e is not None else tokenizer.decode([tid])
            result.append(TokenInfo(token_id=tid, token_str=token_str, start=s, end=e))
        return result

    # Slow tokenizer fallback – reconstruct offsets by walking the text
    result = []
    cursor = 0
    for tid in ids:
        piece = tokenizer.decode([tid])
        # Try to locate piece in the remaining text (handle BPE quirks)
        idx = text.find(piece, cursor)
        if idx == -1:
            # If exact match fails, just assign at cursor with piece length
            idx = cursor
        start = idx
        end = idx + len(piece)
        result.append(TokenInfo(token_id=tid, token_str=piece, start=start, end=end))
        cursor = end
    return result


# --------------- Routes ---------------

@app.post("/api/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    results: list[TokenizerResult] = []
    for name in TOKENIZERS:
        try:
            tok = _get_tokenizer(name)
            tokens = _token_spans(tok, req.text)
            results.append(
                TokenizerResult(
                    model_name=name,
                    num_tokens=len(tokens),
                    num_chars=len(req.text),
                    tokens=tokens,
                )
            )
        except Exception as e:
            # If a tokenizer fails, still return the others
            results.append(
                TokenizerResult(
                    model_name=name,
                    num_tokens=-1,
                    num_chars=len(req.text),
                    tokens=[],
                )
            )
            print(f"Error tokenizing with {name}: {e}")
    return TokenizeResponse(results=results)


@app.get("/api/tokenizers")
async def list_tokenizers():
    """Return the list of configured tokenizer names."""
    return {"tokenizers": TOKENIZERS}


# Serve frontend
FRONTEND = Path(__file__).parent / "index.html"


@app.get("/")
async def root():
    return FileResponse(FRONTEND, media_type="text/html")


# --------------- Startup ---------------

@app.on_event("startup")
async def on_startup():
    preload_tokenizers()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
