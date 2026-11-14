"""Unit and integration tests for the frontend music streaming server."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import queue
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
import wave
from typing import Optional

import pytest

from earth_jukebox.audio import BroadcastAudioBackend, CMinorAudioEngine, NullAudioBackend
from earth_jukebox.cli import parse_args
from earth_jukebox.models import AnalysisResult
from earth_jukebox.server import EarthJukeboxServer, SonificationService, render_sonification_wav
from earth_jukebox.server.app import _EarthJukeboxHTTPServer, _RequestHandler, _ws_encode_frame


def _http_get(url: str, timeout: float = 3.0) -> tuple[int, dict[str, str], bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, dict(resp.headers), resp.read()


def _http_post_json(
    url: str, payload: dict[str, object], timeout: float = 4.0
) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, dict(resp.headers), resp.read()


def _ws_send_masked_text(sock: socket.socket, text: str) -> None:
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", 0x81, 0x80 | length)
    else:
        header = struct.pack("!BBH", 0x81, 0x80 | 126, length)
    masked = bytes(b ^ mask[idx % 4] for idx, b in enumerate(payload))
    sock.sendall(header + mask + masked)


def _ws_read_server_frame(sock: socket.socket) -> tuple[int, bytes]:
    def read_exact(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("WebSocket closed")
            buf.extend(chunk)
        return bytes(buf)

    b1, b2 = read_exact(2)
    opcode = b1 & 0x0F
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", read_exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", read_exact(8))[0]
    payload = read_exact(length) if length > 0 else b""
    return opcode, payload


def test_broadcast_audio_backend_emits_enriched_events_and_cleans_up() -> None:
    backend = BroadcastAudioBackend(inner=NullAudioBackend(), max_queue_size=16)
    sub = backend.subscribe()
    try:
        assert backend.subscriber_count == 1
        backend.start({8: 80})
        backend.note_on(8, 60, 90)
        backend.note_off(8, 60)
        backend.control_change(8, 91, 74)
        backend.close()

        events = []
        while not sub.empty():
            events.append(sub.get_nowait())

        types = [evt["type"] for evt in events]
        assert types == ["engine_start", "note_on", "note_off", "control_change", "engine_stop"]

        note_on_evt = events[1]
        assert note_on_evt["channel"] == 8
        assert note_on_evt["program"] == 80
        assert note_on_evt["midi_note"] == 60
        assert note_on_evt["note_name"] == "C4"
        assert note_on_evt["freq_hz"] == pytest.approx(261.6256, rel=1e-4)
        assert note_on_evt["velocity"] == 90
    finally:
        backend.unsubscribe(sub)
        assert backend.subscriber_count == 0


def test_render_sonification_wav_produces_valid_riff_audio() -> None:
    active = AnalysisResult(
        dominant_color="Red",
        intensity=0.85,
        density=0.78,
        shape="circular",
        shape_score=0.85,
        spottiness=0.55,
        swirliness=0.60,
        sensor_size=64,
        sensor_bbox=(10, 10, 74, 74),
        cursor_xy=(42, 42),
        touching=True,
    )
    wav_active = render_sonification_wav(active, bar_index=0, bars=1, sample_rate=16000)
    with wave.open(io.BytesIO(wav_active), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        frames = wf.readframes(wf.getnframes())
        assert len(frames) > 40000

    idle = AnalysisResult(
        dominant_color="Blue",
        intensity=0.0,
        density=0.0,
        touching=False,
    )
    wav_idle = render_sonification_wav(idle, sample_rate=16000)
    with wave.open(io.BytesIO(wav_idle), "rb") as wf:
        idle_frames = wf.readframes(wf.getnframes())
        assert all(b == 0 for b in idle_frames)


def test_sonification_service_coordinate_resolution_and_validation() -> None:
    service = SonificationService(enable_local_audio=False, ambient_drums=True)
    snap_ratio = service.process_coordinates(
        {"x_ratio": 0.5, "y_ratio": 0.5, "sensor_size": 48, "dataset": 0},
        update_live=False,
    )
    assert snap_ratio["dataset"]["index"] == 0
    assert snap_ratio["coordinates"]["sensor_size"] == 48
    assert snap_ratio["music"]["primary_voice"]["midi_note"] is not None
    assert (
        snap_ratio["music"]["primary_voice"]["midi_note"] % 12
    ) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES

    snap_edge = service.process_coordinates(
        {"x": 99999, "y": -50, "sensor_size": 120},
        update_live=False,
    )
    assert snap_edge["coordinates"]["x"] == snap_edge["dataset"]["width"] - 1
    assert snap_edge["coordinates"]["y"] == 0

    with pytest.raises(ValueError):
        service.process_coordinates({"sensor_size": 40}, update_live=False)


class _InMemorySocket:
    """In-memory socket shim for exercising _RequestHandler without OS TCP ports."""

    def __init__(self, raw_request: bytes) -> None:
        self._rfile = io.BytesIO(raw_request)
        self._wfile = io.BytesIO()

    def makefile(self, mode: str, *_args: object, **_kwargs: object) -> io.BytesIO:
        return self._rfile if "r" in mode else self._wfile

    def sendall(self, data: bytes) -> None:
        self._wfile.write(data)

    def close(self) -> None:
        pass

    @property
    def raw_response(self) -> bytes:
        return self._wfile.getvalue()


def _dispatch_in_memory(
    service: SonificationService,
    method: str,
    path: str,
    body: bytes = b"",
    headers: Optional[dict[str, str]] = None,
) -> tuple[int, dict[str, str], bytes]:
    hdr_lines = [f"{method} {path} HTTP/1.1", "Host: localhost"]
    if body:
        hdr_lines.append("Content-Type: application/json")
        hdr_lines.append(f"Content-Length: {len(body)}")
    if headers:
        for k, v in headers.items():
            hdr_lines.append(f"{k}: {v}")
    raw_req = ("\r\n".join(hdr_lines) + "\r\n\r\n").encode("ascii") + body

    fake_sock = _InMemorySocket(raw_req)
    dummy_server = type(
        "_DummyServer",
        (),
        {"service": service, "stop_event": threading.Event()},
    )()
    _RequestHandler(fake_sock, ("127.0.0.1", 0), dummy_server)  # type: ignore[arg-type]

    head, resp_body = fake_sock.raw_response.split(b"\r\n\r\n", 1)
    lines = head.decode("ascii").split("\r\n")
    status_code = int(lines[0].split(" ", 2)[1])
    resp_headers: dict[str, str] = {}
    for line in lines[1:]:
        if ": " in line:
            k, v = line.split(": ", 1)
            resp_headers[k] = v
    return status_code, resp_headers, resp_body


def test_http_handler_routes_in_memory() -> None:
    """Exercise /, /api/health, /api/datasets, /api/coordinates, /api/audio in-process."""
    service = SonificationService(enable_local_audio=False, ambient_drums=True)
    service.start()
    try:
        st, hdrs, body = _dispatch_in_memory(service, "GET", "/")
        assert st == 200
        assert "text/html" in hdrs["Content-Type"]
        assert b"Earth Jukebox - Coordinate to Music Stream" in body

        st, _h, body = _dispatch_in_memory(service, "GET", "/api/health")
        assert st == 200
        assert json.loads(body.decode("utf-8"))["bpm"] == 75.0

        st, _h, body = _dispatch_in_memory(service, "GET", "/api/datasets")
        assert st == 200
        assert len(json.loads(body.decode("utf-8"))["datasets"]) >= 7

        st, hdrs, body = _dispatch_in_memory(service, "GET", "/api/datasets/0/image")
        assert st == 200
        assert hdrs["Content-Type"] == "image/png"
        assert body.startswith(b"\x89PNG")

        coord_payload = json.dumps(
            {"x_ratio": 0.4, "y_ratio": 0.5, "sensor_size": 48, "dataset": 0}
        ).encode("utf-8")
        st, _h, body = _dispatch_in_memory(service, "POST", "/api/coordinates", body=coord_payload)
        assert st == 200
        snap = json.loads(body.decode("utf-8"))
        assert snap["music"]["primary_voice"]["freq_hz"] > 0

        st, hdrs, body = _dispatch_in_memory(service, "POST", "/api/audio", body=coord_payload)
        assert st == 200
        assert hdrs["Content-Type"] == "audio/wav"
        assert body[:4] == b"RIFF"

        st, hdrs, body = _dispatch_in_memory(
            service, "GET", "/api/audio.wav?x=120&y=90&bars=1&sample_rate=16000"
        )
        assert st == 200
        assert hdrs["Content-Type"] == "audio/wav"
        assert body[:4] == b"RIFF"
    finally:
        service.stop()


@pytest.mark.integration
def test_server_http_wav_sse_and_websocket_coordinate_streaming() -> None:
    server = EarthJukeboxServer(host="127.0.0.1", port=0, enable_local_audio=False)
    server.start()
    try:
        base_url = server.url

        status, _hdrs, html_body = _http_get(f"{base_url}/")
        assert status == 200
        assert b"Earth Jukebox - Coordinate to Music Stream" in html_body

        status, _hdrs, health_raw = _http_get(f"{base_url}/api/health")
        assert status == 200
        health = json.loads(health_raw.decode("utf-8"))
        assert health["status"] == "ok"
        assert health["bpm"] == 75.0

        status, _hdrs, ds_raw = _http_get(f"{base_url}/api/datasets")
        assert status == 200
        datasets = json.loads(ds_raw.decode("utf-8"))["datasets"]
        assert len(datasets) >= 7

        status, img_hdrs, img_bytes = _http_get(f"{base_url}/api/datasets/0/image")
        assert status == 200
        assert img_hdrs.get("Content-Type") == "image/png"
        assert img_bytes.startswith(b"\x89PNG")

        # POST /api/coordinates -> structured C-minor music state
        status, _hdrs, snap_raw = _http_post_json(
            f"{base_url}/api/coordinates",
            {"x_ratio": 0.35, "y_ratio": 0.45, "sensor_size": 56, "dataset": 0},
        )
        assert status == 200
        snap = json.loads(snap_raw.decode("utf-8"))
        assert snap["type"] == "sonification_state"
        assert snap["music"]["primary_voice"]["freq_hz"] > 0
        assert snap["analysis"]["dominant_color"] in {
            "Black", "White", "Red", "Orange", "Green", "Blue"
        }

        # POST /api/audio & GET /api/audio.wav -> playable WAV audio for coordinates
        status, wav_hdrs, wav_bytes = _http_post_json(
            f"{base_url}/api/audio",
            {"x": 240, "y": 180, "bars": 1, "sample_rate": 16000},
        )
        assert status == 200
        assert wav_hdrs.get("Content-Type") == "audio/wav"
        assert "X-EarthJukebox-Chord" in wav_hdrs
        assert wav_bytes[:4] == b"RIFF"

        status, _hdrs, wav_get_bytes = _http_get(
            f"{base_url}/api/audio.wav?x_ratio=0.6&y_ratio=0.4&bars=1&sample_rate=16000"
        )
        assert status == 200
        assert wav_get_bytes[:4] == b"RIFF"

        # GET /api/stream (SSE) receives connected + live music note_on events
        sse_events: list[str] = []
        with urllib.request.urlopen(f"{base_url}/api/stream", timeout=3.0) as sse_resp:
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline and "note_on" not in sse_events:
                line = sse_resp.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("event: "):
                    sse_events.append(line.removeprefix("event: "))
        assert "connected" in sse_events
        assert "note_on" in sse_events

        # GET /ws (RFC 6455 WebSocket) sends coordinates and receives state + music events
        host, port = server.address
        with socket.create_connection((host, port), timeout=3.0) as ws_sock:
            ws_key = base64.b64encode(b"earth-jukebox-ws").decode("ascii")
            handshake = (
                f"GET /ws HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {ws_key}\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii")
            ws_sock.sendall(handshake)

            raw_hdr = bytearray()
            while b"\r\n\r\n" not in raw_hdr:
                raw_hdr.extend(ws_sock.recv(1024))
            hdr_text = raw_hdr.split(b"\r\n\r\n", 1)[0].decode("ascii")
            expected_accept = base64.b64encode(
                hashlib.sha1(
                    (ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                ).digest()
            ).decode("ascii")
            assert "101 Switching Protocols" in hdr_text
            assert expected_accept in hdr_text

            _ws_send_masked_text(
                ws_sock,
                json.dumps({"x_ratio": 0.42, "y_ratio": 0.52, "sensor_size": 48, "touching": True}),
            )

            received_types: set[str] = set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and (
                "sonification_state" not in received_types or "note_on" not in received_types
            ):
                opcode, payload = _ws_read_server_frame(ws_sock)
                if opcode == 0x1:
                    msg = json.loads(payload.decode("utf-8"))
                    received_types.add(str(msg.get("type")))

            assert "connected" in received_types
            assert "sonification_state" in received_types
            assert "note_on" in received_types
    finally:
        server.stop()


def test_cli_parse_args_serve_flags() -> None:
    parsed = parse_args(["--serve", "--host", "0.0.0.0", "--port", "9000", "--local-audio"])
    assert parsed.serve is True
    assert parsed.host == "0.0.0.0"
    assert parsed.port == 9000
    assert parsed.local_audio is True
