"""Drive the backend's realtime WebSocket and report what the segmented decoder streams.

Sends a clip as timed appends, commits, and collects transcription.delta / transcription.done. Reports
the delta count and arrival offsets (does it stream during the turn?), whether concatenated deltas equal
the final text (retraction check), and WER against the one-shot /v1/audio/transcriptions baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave

import httpx
import websockets

sys.path.insert(0, "/private/tmp/claude-501/-Users-ieaves-repos-sprag-dialtone/0442b38a-7339-4103-998f-16c18a27240d/scratchpad")
from incremental_probe import _wer  # noqa: E402


def load_pcm16(path: str, seconds: float | None) -> bytes:
    """Read a 16 kHz mono PCM16 wav, optionally truncated to ``seconds``."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit(f"{path}: expected 16 kHz mono")
        raw = w.readframes(w.getnframes())
    return raw[: int(seconds * 16000 * 2)] if seconds else raw


async def run(url: str, model: str, pcm: bytes, append_ms: int, realtime: bool) -> dict:
    """Stream ``pcm`` over the realtime socket and collect the transcript events."""
    deltas: list[tuple[float, str]] = []
    done: dict | None = None
    chunk = int(16000 * 2 * append_ms / 1000)
    async with websockets.connect(f"{url}/v1/realtime?model={model}", open_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        # A non-final commit opens the generation; the final one closes the audio stream.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        start = time.monotonic()

        async def reader():
            nonlocal done
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "transcription.delta":
                    deltas.append((time.monotonic() - start, event.get("delta") or ""))
                elif kind == "transcription.done":
                    done = event
                    return
                elif kind == "error":
                    done = {"error": event}
                    return

        task = asyncio.create_task(reader())
        for i in range(0, len(pcm), chunk):
            await ws.send(
                json.dumps(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + chunk]).decode()}
                )
            )
            if realtime:
                await asyncio.sleep(append_ms / 1000)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        commit_at = time.monotonic() - start
        try:
            await asyncio.wait_for(task, timeout=180)
        except TimeoutError:
            task.cancel()
    return {"deltas": deltas, "done": done, "commit_at": commit_at, "audio_s": len(pcm) / 32000}


def one_shot(url: str, model: str, pcm: bytes) -> str:
    """Transcribe the whole clip via the batch route, as the quality baseline."""
    buf = wave.open(out := __import__("io").BytesIO(), "wb")
    buf.setnchannels(1)
    buf.setsampwidth(2)
    buf.setframerate(16000)
    buf.writeframes(pcm)
    buf.close()
    r = httpx.post(
        f"{url}/v1/audio/transcriptions",
        files={"file": ("c.wav", out.getvalue(), "audio/wav")},
        data={"model": model},
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def main() -> None:
    """Run one permutation and print its measurements."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:18098")
    ap.add_argument("--ws", default="ws://localhost:18098")
    ap.add_argument("--model", default="rhythm")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--append-ms", type=int, default=200)
    ap.add_argument("--realtime", action="store_true", help="pace appends in real time (else dump)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pcm = load_pcm16(args.clip, args.seconds)
    base = one_shot(args.url, args.model, pcm)
    result = asyncio.run(run(args.ws, args.model, pcm, args.append_ms, args.realtime))
    done = result["done"] or {}
    if "error" in done:
        print(json.dumps({"label": args.label, "error": done["error"]}, ensure_ascii=False))
        return
    text = (done.get("text") or "").strip()
    joined = "".join(d for _, d in result["deltas"]).strip()
    offsets = [round(t, 1) for t, _ in result["deltas"]]
    print(
        json.dumps(
            {
                "label": args.label,
                "audio_s": round(result["audio_s"], 1),
                "deltas": len(result["deltas"]),
                "first_delta_s": offsets[0] if offsets else None,
                "commit_at_s": round(result["commit_at"], 1),
                "streamed_during_turn": bool(offsets) and offsets[0] < result["commit_at"] - 0.2,
                "deltas_equal_final": joined == text,
                "wer_vs_one_shot": round(_wer(base, text), 3),
                "usage": done.get("usage"),
                "one_shot": base[:110],
                "final": text[:110],
                "joined_deltas": joined[:110],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
