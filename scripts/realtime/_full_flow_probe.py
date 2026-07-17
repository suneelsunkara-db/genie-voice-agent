"""Drive the FULL realtime API over the live WebSocket with a recorded WAV.

Streams real captured audio (the exact spoken sentence) through STT -> LLM ->
streaming TTS and prints the transcript, the assistant reply, and audio stats.

    python scripts/realtime/_full_flow_probe.py /tmp/realtime_audio/b9a8d155_turn1.wav
"""
from __future__ import annotations

import asyncio
import json
import sys
import wave

import websockets

WS_URL = "ws://localhost:8001/v1/realtime/voice"
FRAME_BYTES = 1280  # 40ms @ 16kHz s16le


async def run(wav_path: str) -> None:
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    print(f"wav={wav_path}  sr={sr}  bytes={len(pcm)}  dur={len(pcm)/2/sr:.1f}s\n")

    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "language": "en-US",
                                  "sample_rate_hz": sr, "encoding": "pcm_s16le"}))
        # wait for session.ready
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "session.ready":
                break

        # stream the audio, then force-finalize with audio.end
        for i in range(0, len(pcm), FRAME_BYTES):
            await ws.send(pcm[i:i + FRAME_BYTES])
            await asyncio.sleep(0.005)
        await ws.send(json.dumps({"type": "audio.end"}))

        transcript = reply = None
        audio_chunks = audio_bytes = 0
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                t = msg.get("type")
                if t == "transcript.final":
                    transcript = msg.get("text")
                    print(f"TRANSCRIPT ({msg.get('language')}): {transcript}\n")
                elif t == "response.text":
                    reply = msg.get("text")
                    print(f"ASSISTANT REPLY: {reply}\n")
                elif t == "response.audio":
                    audio_chunks += 1
                    audio_bytes += len(msg.get("audio_b64") or "")
                    if msg.get("final"):
                        print(f"AUDIO: {audio_chunks} chunks, ~{audio_bytes} b64 bytes "
                              f"(server_ttfb_ms={msg.get('server_ttfb_ms')})")
                        break
                elif t == "error":
                    print(f"ERROR: {msg}")
                    break
        except asyncio.TimeoutError:
            print("timed out waiting for events")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/realtime_audio/b9a8d155_turn1.wav"
    asyncio.run(run(path))
