"""Drive the realtime socket one generation per segment: commit, feed a segment, await its done, repeat.

Each generation carries exactly one audio, which is what the model's realtime processor asserts. The
client holds the audio and paces it, because the driver clears its queue after every completed turn.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import wave

import httpx
import websockets

sys.path.insert(0, "/private/tmp/claude-501/-Users-ieaves-repos-sprag-dialtone/0442b38a-7339-4103-998f-16c18a27240d/scratchpad")
from incremental_probe import _wer  # noqa: E402


def load(path, seconds):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return raw[: int(seconds * 16000 * 2)] if seconds else raw


def wav(pcm):
    b = io.BytesIO()
    w = wave.open(b, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(pcm)
    w.close()
    return b.getvalue()


def one_shot(url, model, pcm):
    r = httpx.post(
        f"{url}/v1/audio/transcriptions",
        files={"file": ("c.wav", wav(pcm), "audio/wav")},
        data={"model": model},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


async def run(ws_url, model, pcm, segment_s, append_ms):
    """Feed the clip a segment at a time, awaiting each segment's transcription.done."""
    seg_bytes = int(segment_s * 16000 * 2)
    chunk = int(16000 * 2 * append_ms / 1000)
    texts, timings, deltas = [], [], []
    async with websockets.connect(f"{ws_url}/v1/realtime?model={model}", open_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        start = time.monotonic()
        for off in range(0, len(pcm), seg_bytes):
            segment = pcm[off : off + seg_bytes]
            final = off + seg_bytes >= len(pcm)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
            for i in range(0, len(segment), chunk):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(segment[i : i + chunk]).decode(),
                        }
                    )
                )
            if final:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            sent_at = time.monotonic()
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if event.get("type") == "transcription.delta":
                    deltas.append(event.get("delta") or "")
                    continue
                if event.get("type") == "transcription.done":
                    texts.append((event.get("text") or "").strip())
                    timings.append(time.monotonic() - sent_at)
                    break
                if event.get("type") == "error":
                    return texts, timings, event, time.monotonic() - start, deltas
    return texts, timings, None, time.monotonic() - start, deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:18098")
    ap.add_argument("--ws", default="ws://localhost:18098")
    ap.add_argument("--model", default="rhythm")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--segment-s", type=float, default=8.0)
    ap.add_argument("--append-ms", type=int, default=200)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pcm = load(args.clip, args.seconds)
    try:
        base = one_shot(args.url, args.model, pcm)
    except Exception:
        base = ""  # the batch route caps duration; the realtime path is still measurable
    texts, timings, err, wall, deltas = asyncio.run(run(args.ws, args.model, pcm, args.segment_s, args.append_ms))
    if err:
        print(json.dumps({"label": args.label, "error": str(err)[:200]}))
        return
    joined = " ".join(t for t in texts if t)
    empties = sum(1 for t in texts if not t)
    print(
        json.dumps(
            {
                "label": args.label,
                "audio_s": round(len(pcm) / 32000, 1),
                "segments": len(texts),
                "n_deltas": len(deltas),
                "deltas_match_dones": "".join(deltas).split() == " ".join(t for t in texts if t).split(),
                "empty_segments": empties,
                "words": len(joined.split()),
                "one_shot_words": len(base.split()),
                "wer_vs_one_shot": round(_wer(base, joined), 3) if base else None,
                "segment_latency_p50_ms": round(1000 * sorted(timings)[len(timings) // 2], 1) if timings else None,
                "wall_s": round(wall, 1),
                "joined_deltas": "".join(deltas),
                "final": joined,
                "one_shot": base,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
