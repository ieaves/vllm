# Realtime probes

Harnesses used to measure the realtime seam against a live Qwen3-ASR deployment.

- `ws_raw.py` — one commit, stream audio, dump every delta verbatim plus a per-segment breakdown
- `ws_probe.py` — same, scored: WER against `/v1/audio/transcriptions`, delta/done consistency
- `ws_seg.py` — one generation per segment (commit, feed, await done, repeat)
- `boundary.py` — counts characters duplicated at segment joins
- `incremental_probe.py` — the chat-completions path (window + forced prefix + rollback)

All take `--clip <16kHz mono wav>` and point at `http://localhost:18098` by default, so they
expect a port-forward to the model server. The clip used for the reported measurements was a
real customer voice sample and is deliberately not committed here.
