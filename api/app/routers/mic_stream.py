"""Browser mic -> Deepgram live STT WebSocket proxy.

Keeps the API key server-side while the UI streams PCM audio chunks and
receives interim/final transcripts in real time.
"""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlencode

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from genie_voice.config import get_settings

router = APIRouter(tags=["mic-stream"])


def _deepgram_listen_url(sample_rate: int) -> str:
    params = urlencode(
        {
            "model": "nova-3",
            "smart_format": "true",
            "punctuate": "true",
            "interim_results": "true",
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": "1",
        }
    )
    return f"wss://api.deepgram.com/v1/listen?{params}"


@router.websocket("/calls/{call_id}/mic-stream")
async def mic_stream(websocket: WebSocket, call_id: str) -> None:
    await websocket.accept()
    settings = get_settings()
    key = settings.secrets.deepgram_api_key.strip()
    if not key:
        await websocket.send_json({"type": "error", "message": "DEEPGRAM_API_KEY is not configured"})
        await websocket.close()
        return

    sample_rate = 16000
    if websocket.query_params.get("sample_rate"):
        try:
            sample_rate = int(websocket.query_params["sample_rate"])
        except ValueError:
            sample_rate = 16000

    import websockets

    dg_url = _deepgram_listen_url(sample_rate)
    headers = {"Authorization": f"Token {key}"}

    try:
        async with websockets.connect(dg_url, additional_headers=headers) as dg_ws:

            async def client_to_deepgram() -> None:
                try:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            await dg_ws.send(json.dumps({"type": "CloseStream"}))
                            break
                        if message.get("bytes"):
                            await dg_ws.send(message["bytes"])
                        elif message.get("text"):
                            payload = json.loads(message["text"])
                            if payload.get("type") == "stop":
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                                break
                except WebSocketDisconnect:
                    try:
                        await dg_ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception:  # noqa: BLE001
                        pass

            async def deepgram_to_client() -> None:
                async for raw in dg_ws:
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    msg_type = payload.get("type")
                    if msg_type == "Results":
                        channel = payload.get("channel") or {}
                        alts = channel.get("alternatives") or []
                        transcript = ((alts[0] or {}).get("transcript") if alts else "") or ""
                        await websocket.send_json(
                            {
                                "type": "transcript",
                                "call_id": call_id,
                                "transcript": transcript.strip(),
                                "is_final": bool(payload.get("is_final")),
                                "speech_final": bool(payload.get("speech_final")),
                            }
                        )
                    elif msg_type == "Error":
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": payload.get("message") or payload.get("description") or "Deepgram error",
                            }
                        )

            # Run both directions; let Deepgram drain finals after CloseStream.
            await asyncio.gather(client_to_deepgram(), deepgram_to_client())
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        try:
            await websocket.send_json({"type": "stream_end"})
        except Exception:  # noqa: BLE001
            pass
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
