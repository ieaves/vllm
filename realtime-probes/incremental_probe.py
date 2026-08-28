"""Validate the incremental (streaming) ASR algorithm against a live rhythm pod, no backend change.

Drives the deployed ``rhythm`` chat-completions continuation endpoint with the Qwen3-ASR reference
streaming loop (growing/bounded window, forced decoded prefix on the assistant turn, token rollback) and
compares the streamed transcript to the one-shot ``/v1/audio/transcriptions`` baseline on the natural-speech
clips. Answers, before any seed change, whether the approach matches one-shot quality and at what parameters.

Run against a port-forwarded pod (see the printed usage), from inside ``gateway/`` so ``gateway.audio``
imports:

    kubectl -n inference port-forward svc/rhythm-g4-model-server 18098:8000   # in one shell
    uv run python <this>.py --url http://localhost:18098 \
        --clips /Users/ieaves/repos/sprag/test1-hero-natural.wav ...
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import time
import wave
from dataclasses import dataclass, field

import httpx
import numpy as np
from tokenizers import Tokenizer

from gateway.audio import pcm16_to_wav, resample_pcm16

_RATE = 16000
_LANG_NAMES = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese"}
_TAG = "<asr_text>"


def _load_tokenizer() -> Tokenizer:
    """Load the cached Qwen3 tokenizer from the local HF cache, no network."""
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(f"{home}/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json")
    if not hits:
        raise SystemExit("Qwen3 tokenizer.json not in HF cache; warm it first")
    return Tokenizer.from_file(hits[0])


def _load_pcm16_16k(path: str) -> bytes:
    """Read a WAV file to mono little-endian PCM16 at 16 kHz."""
    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    pcm = samples.tobytes()
    return resample_pcm16(pcm, rate, _RATE)


def _rollback(tok: Tokenizer, text: str, k: int) -> str:
    """Drop the last ``k`` tokens from ``text``, widening past a token cut that lands mid-codepoint."""
    ids = tok.encode(text, add_special_tokens=False).ids
    while k <= len(ids):
        end = len(ids) - k
        prefix = tok.decode(ids[:end]) if end > 0 else ""
        if "�" not in prefix:
            return prefix
        if end == 0:
            return ""
        k += 1
    return ""


def _wer(ref: str, hyp: str) -> float:
    """Word error rate of ``hyp`` against ``ref`` (Levenshtein over whitespace words)."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / max(1, len(r))


@dataclass
class Params:
    """One incremental configuration to evaluate."""

    segment_s: float = field(default=1.0)
    max_window_s: float = field(default=20.0)
    rollback_tokens: int = field(default=5)
    unfixed_segments: int = field(default=2)
    hold_back_words: int = field(default=5)
    language: str = field(default="en")
    max_tokens: int = field(default=256)


def _one_shot(client: httpx.Client, url: str, model: str, pcm: bytes) -> str:
    """Transcribe the whole clip via /v1/audio/transcriptions — the quality baseline."""
    wav = pcm16_to_wav(pcm, _RATE)
    resp = client.post(
        f"{url}/v1/audio/transcriptions",
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"model": model},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def _pass(
    client: httpx.Client, url: str, model: str, window: bytes, prefix: str, header: str, max_tokens: int
) -> tuple[str, int, int, float]:
    """Run one continuation pass; return (committed transcript, completion tokens, prompt tokens, wall seconds).

    Reconstructs the transcript robustly across ``continue_final_message`` variants: the response may carry
    only the generated continuation, or echo the whole assistant message (header + prefix + continuation).
    """
    wav_b64 = base64.b64encode(pcm16_to_wav(window, _RATE)).decode("ascii")
    messages = [
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}]},
        {"role": "assistant", "content": header + prefix},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    t0 = time.monotonic()
    resp = client.post(f"{url}/v1/chat/completions", json=body, timeout=120)
    dt = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    content = (data["choices"][0]["message"]["content"] or "").strip()
    if _TAG in content:  # echoed the whole message: header + prefix + tail
        committed = content.rsplit(_TAG, 1)[1].strip()
    elif prefix and content.startswith(prefix):  # echoed prefix + tail without the header
        committed = content
    elif prefix and content[:1] in ",.?!;:":  # continuation opens with punctuation: no joining space
        committed = f"{prefix}{content}"
    else:  # continuation only
        committed = f"{prefix} {content}".strip() if prefix else content
    usage = data.get("usage", {})
    return committed, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0), dt


def run_clip(client: httpx.Client, url: str, model: str, path: str, p: Params) -> dict:
    """Evaluate one clip: one-shot baseline vs the incremental loop under ``p``."""
    pcm = _load_pcm16_16k(path)
    total_s = len(pcm) / (_RATE * 2)
    tok = _load_tokenizer()
    lang_name = _LANG_NAMES.get(p.language, p.language)
    header = f"language {lang_name}{_TAG}"

    one_shot = _one_shot(client, url, model, pcm)

    seg_bytes = int(p.segment_s * _RATE * 2)
    win_bytes = int(p.max_window_s * _RATE * 2)
    committed = ""
    emitted_words: list[str] = []
    retractions = 0
    latencies: list[float] = []
    gen_tokens = 0
    prompt_tokens_total = 0
    first_emit_s: float | None = None
    n_passes = 0

    end = seg_bytes
    while True:
        final = end >= len(pcm)
        cur_end = min(end, len(pcm))
        window = pcm[max(0, cur_end - win_bytes) : cur_end]
        prefix = "" if n_passes < p.unfixed_segments else _rollback(tok, committed, p.rollback_tokens)
        committed, tokens, prompt_toks, dt = _pass(client, url, model, window, prefix, header, p.max_tokens)
        latencies.append(dt)
        gen_tokens += tokens
        prompt_tokens_total += prompt_toks
        n_passes += 1

        words = committed.split()
        boundary = len(words) if final else max(len(emitted_words), len(words) - p.hold_back_words)
        boundary = min(boundary, len(words))
        new_emitted = words[:boundary]
        for i in range(min(len(emitted_words), len(new_emitted))):
            if emitted_words[i] != new_emitted[i]:
                retractions += 1
                break
        if first_emit_s is None and boundary > 0:
            first_emit_s = cur_end / (_RATE * 2)
        emitted_words = new_emitted
        if final:
            break
        end += seg_bytes

    final_text = " ".join(emitted_words)
    return {
        "clip": os.path.basename(path),
        "audio_s": round(total_s, 1),
        "passes": n_passes,
        "one_shot": one_shot,
        "incremental": final_text,
        "wer_vs_one_shot": round(_wer(one_shot, final_text), 4),
        "retractions": retractions,
        "first_emit_audio_s": first_emit_s,
        "completion_tokens_total": gen_tokens,
        "prompt_tokens_total": prompt_tokens_total,
        "pass_ms_p50": round(1000 * float(np.percentile(latencies, 50)), 1),
        "pass_ms_max": round(1000 * max(latencies), 1),
    }


def main() -> None:
    """Run the incremental probe over every clip and print one JSON summary per clip."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:18098")
    ap.add_argument("--model", default="rhythm")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--segment-s", type=float, default=1.0)
    ap.add_argument("--max-window-s", type=float, default=20.0)
    ap.add_argument("--rollback-tokens", type=int, default=5)
    ap.add_argument("--unfixed-segments", type=int, default=2)
    ap.add_argument("--hold-back-words", type=int, default=5)
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    p = Params(
        segment_s=args.segment_s,
        max_window_s=args.max_window_s,
        rollback_tokens=args.rollback_tokens,
        unfixed_segments=args.unfixed_segments,
        hold_back_words=args.hold_back_words,
        language=args.language,
        max_tokens=args.max_tokens,
    )
    print(f"params: {p}\n")
    with httpx.Client() as client:
        for path in args.clips:
            try:
                result = run_clip(client, args.url, args.model, path, p)
            except Exception as exc:  # noqa: BLE001 - a probe: report and continue to the next clip
                print(json.dumps({"clip": os.path.basename(path), "error": repr(exc)}))
                continue
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
