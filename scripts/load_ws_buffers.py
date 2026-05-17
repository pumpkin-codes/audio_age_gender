#!/usr/bin/env python3
"""Load test: concurrent WebSocket streams with rolling PCM buffers.

Example (server must be up with VOICE_LOAD_TEST_SKIP_INFERENCE=1 for soak):
  python scripts/load_ws_buffers.py --connections 1000 --ramp 100 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import websockets

# 100 ms of silence @ 16 kHz mono s16le (matches ws_client chunk sizing)
PCM_CHUNK = b"\x00" * 3200


@dataclass
class LoadStats:
    connected: int = 0
    failed: int = 0
    ready_ms: list[float] = field(default_factory=list)
    partials: int = 0
    errors: list[str] = field(default_factory=list)

    def record_error(self, msg: str, limit: int = 20) -> None:
        if len(self.errors) < limit:
            self.errors.append(msg)


async def stream_client(
    client_id: int,
    uri: str,
    duration_s: float,
    chunk_interval_s: float,
    stats: LoadStats,
) -> None:
    contact_id = f"load-{client_id:05d}"
    try:
        async with websockets.connect(
            uri,
            open_timeout=60,
            close_timeout=5,
            max_size=2**20,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "config", "contact_id": contact_id}))
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            ready = json.loads(raw)
            if ready.get("type") != "ready":
                raise RuntimeError(f"unexpected ready payload: {raw[:200]}")
            stats.ready_ms.append((time.perf_counter() - t0) * 1000)
            stats.connected += 1

            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                await ws.send(PCM_CHUNK)
                await asyncio.sleep(chunk_interval_s)
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    payload = json.loads(msg)
                    if payload.get("type") == "partial":
                        stats.partials += 1
                except asyncio.TimeoutError:
                    pass

            await ws.send(json.dumps({"type": "end"}))
            try:
                await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                pass
    except Exception as exc:
        stats.failed += 1
        stats.record_error(f"client {client_id}: {type(exc).__name__}: {exc}")


async def run_load(
    uri: str,
    connections: int,
    ramp_per_sec: float,
    duration_s: float,
    chunk_interval_s: float,
) -> LoadStats:
    stats = LoadStats()
    tasks: list[asyncio.Task[None]] = []
    delay = 1.0 / ramp_per_sec if ramp_per_sec > 0 else 0.0

    print(
        f"Ramping {connections} connections @ {ramp_per_sec}/s, "
        f"holding {duration_s}s, chunk every {chunk_interval_s}s → {uri}"
    )

    for i in range(connections):
        tasks.append(
            asyncio.create_task(
                stream_client(i, uri, duration_s, chunk_interval_s, stats)
            )
        )
        if delay > 0 and i < connections - 1:
            await asyncio.sleep(delay)

    await asyncio.gather(*tasks, return_exceptions=True)
    return stats


def print_report(stats: LoadStats, elapsed_s: float, connections: int) -> int:
    print("\n--- load test report ---")
    print(f"target connections: {connections}")
    print(f"connected:          {stats.connected}")
    print(f"failed:             {stats.failed}")
    print(f"partials received:  {stats.partials}")
    print(f"wall time:          {elapsed_s:.1f}s")

    if stats.ready_ms:
        print(
            f"ready latency ms:   "
            f"p50={statistics.median(stats.ready_ms):.0f} "
            f"p95={_percentile(stats.ready_ms, 95):.0f} "
            f"max={max(stats.ready_ms):.0f}"
        )

    if stats.errors:
        print("sample errors:")
        for err in stats.errors:
            print(f"  - {err}")

    success_rate = stats.connected / connections if connections else 0.0
    print(f"success rate:       {success_rate * 100:.2f}%")

    if stats.failed > 0 or stats.connected < connections:
        return 1
    return 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(len(ordered) * pct / 100)
    idx = min(idx, len(ordered) - 1)
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket concurrent stream load test")
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws/stream")
    parser.add_argument("--connections", type=int, default=1000)
    parser.add_argument("--ramp", type=float, default=100.0, help="connections per second")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds per connection")
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=0.2,
        help="seconds between PCM chunks (default 200ms)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    stats = asyncio.run(
        run_load(
            args.uri,
            args.connections,
            args.ramp,
            args.duration,
            args.chunk_interval,
        )
    )
    elapsed = time.perf_counter() - t0
    raise SystemExit(print_report(stats, elapsed, args.connections))


if __name__ == "__main__":
    main()
