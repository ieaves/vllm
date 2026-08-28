"""Measure how often a segment boundary duplicates a character in the delta stream.

Collects each delta separately, then inspects every join point: a boundary is flagged when
the delta's first character repeats the last character already delivered, and one-shot
transcription of the same clip does not contain that doubled pair.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import time
import wave

import httpx
import websockets


def load(path, seconds):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return raw[: int(seconds * 16000 * 2)] if seconds else raw


def one_shot(url, model, pcm):
    b = io.BytesIO()
    w = wave.open(b, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(pcm)
    w.close()
    r = httpx.post(
        f"{url}/v1/audio/transcriptions",
        files={"file": ("c.wav", b.getvalue(), "audio/wav")},
        data={"model": model},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


async def collect(ws_url, model, pcm, append_ms, pace):
    """Stream the clip through one generation and return every delta in order."""
    deltas, done = [], None
    chunk = int(16000 * 2 * append_ms / 1000)
    async with websockets.connect(f"{ws_url}/v1/realtime?model={model}", open_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

        async def reader():
            nonlocal done
            while True:
                e = json.loads(await ws.recv())
                if e.get("type") == "transcription.delta":
                    deltas.append(e.get("delta") or "")
                elif e.get("type") in ("transcription.done", "error"):
                    done = e
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
    return deltas, done


def analyse(deltas, baseline):
    """Return (boundaries, doubled) for the delta stream against a one-shot baseline."""
    norm = re.sub(r"\s+", " ", baseline)
    acc, boundaries, doubled = "", 0, []
    for d in deltas:
        if not d:
            continue
        if acc:
            boundaries += 1
            prev, first = acc[-1], d[0]
            if prev == first and prev.strip():
                pair = prev + first
                # a genuine double letter shows up in the one-shot text too
                window = (acc[-12:] + d[:12]).strip()
                spurious = pair not in norm or window.replace(pair, prev, 1) in norm
                doubled.append((boundaries, repr(acc[-14:] + "|" + d[:14]), spurious))
        acc += d
    return boundaries, doubled, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:18098")
    ap.add_argument("--ws", default="ws://localhost:18098")
    ap.add_argument("--model", default="rhythm")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--append-ms", type=int, default=200)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pcm = load(args.clip, args.seconds)
    base = one_shot(args.url, args.model, pcm)
    for t in range(args.trials):
        deltas, done = collect(args.ws, args.model, pcm, args.append_ms, True) if False else asyncio.run(
            collect(args.ws, args.model, pcm, args.append_ms, True)
        )
        if done and done.get("type") == "error":
            print(f"trial {t + 1}: ERROR {str(done)[:150]}")
            continue
        boundaries, doubled, acc = analyse(deltas, base)
        spurious = [d for d in doubled if d[2]]
        print(
            f"trial {t + 1}: {args.seconds:.0f}s | deltas={len(deltas)} boundaries={boundaries} "
            f"doubled={len(doubled)} spurious={len(spurious)} "
            f"rate={len(spurious) / boundaries:.2f}" if boundaries else "no boundaries"
        )
        if args.show:
            for idx, ctx, sp in doubled:
                print(f"    boundary {idx:>3} {'SPURIOUS' if sp else 'genuine '} {ctx}")


if __name__ == "__main__":
    main()
