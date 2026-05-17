"""WebSocket /ws/stream — progressive predictions."""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.stream_session import StreamSession

router = APIRouter()


@router.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    service = websocket.app.state.analysis_service
    session: StreamSession | None = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")

                if msg_type == "config":
                    session = StreamSession(service, payload.get("contact_id"))
                    await websocket.send_json(
                        {"type": "ready", "contact_id": session.contact_id}
                    )
                    continue

                if msg_type == "end":
                    if session is None:
                        session = StreamSession(service)
                    result = session.analyze_buffer(final=True)
                    if result:
                        await websocket.send_json(result.model_dump())
                    session.clear()
                    continue

            if "bytes" in message and message["bytes"]:
                if session is None:
                    session = StreamSession(service)
                chunk = message["bytes"]
                # Try WAV decode; fall back to PCM s16le
                if chunk[:4] == b"RIFF" or chunk[:4] == b"fLaC" or chunk[:3] == b"ID3":
                    session.append_wav_bytes(chunk)
                else:
                    session.append_pcm16le(chunk)

                if session.should_emit_partial():
                    partial = session.analyze_buffer(final=False)
                    if partial:
                        await websocket.send_json(partial.model_dump())

    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            session.clear()
