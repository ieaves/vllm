"""Drive the realtime socket with a single commit and dump every delta verbatim.

Reports the full raw stream, per-segment breakdown (split on the model's answer header), and delta
arrival offsets, so a segment that decodes to nothing is visible as an empty span rather than hidden
inside an aggregate score.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
import wave

import websockets


def load(path, seconds):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return raw[: int(seconds * 16000 * 2)] if seconds else raw


async def run(ws_url, model, pcm, append_ms, pace):
    """Open one generation, stream the clip into it, and collect every event."""
    deltas, done, err = [], None, None
    chunk = int(16000 * 2 * append_ms / 1000)
    async with websockets.connect(f"{ws_url}/v1/realtime?model={model}", open_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        start = time.monotonic()

        async def reader():
            nonlocal done, err
            while True:
                event = json.loads(await ws.recv())
                kind = event.get("type")
                if kind == "transcription.delta":
                    deltas.append((round(time.monotonic() - start, 2), event.get("delta") or ""))
                elif kind == "transcription.done":
                    done = event
                    return
                elif kind == "error":
                    err = event
                    return

        task = asyncio.create_task(reader())
        for i in range(0, len(pcm), chunk):
            await ws.send(
                json.dumps(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + chunk]).decode()}
                )
            )
            if pace:
                await asyncio.sleep(append_ms / 1000)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        try:
            await asyncio.wait_for(task, timeout=240)
        except TimeoutError:
            task.cancel()
    return deltas, done, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ws", default="ws://localhost:18098")
    ap.add_argument("--model", default="rhythm")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--append-ms", type=int, default=200)
    ap.add_argument("--no-pace", action="store_true", help="dump audio as fast as possible")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pcm = load(args.clip, args.seconds)
    deltas, done, err = asyncio.run(run(args.ws, args.model, pcm, args.append_ms, not args.no_pace))
    if err:
        print(f"ERROR: {str(err)[:300]}")
        return
    raw = "".join(d for _, d in deltas)
    done = done or {}
    usage = done.get("usage") or {}
    print(f"--- {args.label} | audio {len(pcm)/32000:.1f}s ---")
    print(f"deltas={len(deltas)} first={deltas[0][0] if deltas else None}s last={deltas[-1][0] if deltas else None}s")
    print(f"usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
    print(f"raw length: {len(raw)} chars")
    # Each segment restates the answer header, so headers mark segment boundaries in the raw stream.
    parts = re.split(r"(language [^<\n]{0,40}<asr_text>)", raw)
    segments = []
    for i in range(1, len(parts), 2):
        segments.append((parts[i], parts[i + 1] if i + 1 < len(parts) else ""))
    print(f"segments (by header): {len(segments)}")
    for i, (header, body) in enumerate(segments):
        body = body.strip()
        flag = "  <-- EMPTY" if not body else ""
        print(f"  [{i:>2}] {header!r} -> {len(body):>4} chars{flag}  {body[:90]!r}")
    if not segments:
        print("  (no headers found) raw:", raw[:400])
    print("FULL RAW:")
    print(raw)


if __name__ == "__main__":
    main()
