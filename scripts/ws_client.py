#!/usr/bin/env python3
"""WebSocket smoke client for /ws/stream."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets


async def run(uri: str, wav_path: Path, chunk_ms: int = 200) -> None:
    data = wav_path.read_bytes()
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "config", "contact_id": "ws-smoke-test"}))
        print(await ws.recv())

        # Send WAV in chunks (server accumulates RIFF stream; PCM s16le also supported)
        chunk_size = 3200
        for i in range(0, len(data), chunk_size):
            await ws.send(data[i : i + chunk_size])
            await asyncio.sleep(chunk_ms / 1000)
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                print(msg)
            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"type": "end"}))
        print(await ws.recv())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, nargs="?", default=Path("samples/test_clip.wav"))
    parser.add_argument("--uri", default="ws://localhost:8000/ws/stream")
    args = parser.parse_args()
    asyncio.run(run(args.uri, args.wav))


if __name__ == "__main__":
    main()
