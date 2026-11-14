"""Pure-stdlib HTTP, SSE, WAV, and RFC 6455 WebSocket server for Earth Jukebox."""

from __future__ import annotations

import base64
import hashlib
import json
import queue
import socket
import struct
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from earth_jukebox.server.frontend_html import FRONTEND_DEMO_HTML
from earth_jukebox.server.service import SonificationService

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_encode_frame(opcode: int, payload: bytes) -> bytes:
    """Encode an unmasked server-to-client RFC 6455 WebSocket frame."""
    first_byte = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", first_byte, length)
    elif length < 65536:
        header = struct.pack("!BBH", first_byte, 126, length)
    else:
        header = struct.pack("!BBQ", first_byte, 127, length)
    return header + payload


def _ws_read_exact(sock: socket.socket, size: int) -> Optional[bytes]:
    chunks = bytearray()
    while len(chunks) < size:
        try:
            chunk = sock.recv(size - len(chunks))
        except OSError:
            return None
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _ws_read_frame(sock: socket.socket) -> Optional[tuple[int, bytes]]:
    """Decode one client-to-server RFC 6455 WebSocket frame."""
    header = _ws_read_exact(sock, 2)
    if header is None:
        return None
    b1, b2 = header[0], header[1]
    opcode = b1 & 0x0F
    masked = (b2 & 0x80) != 0
    length = b2 & 0x7F

    if length == 126:
        ext = _ws_read_exact(sock, 2)
        if ext is None:
            return None
        length = struct.unpack("!H", ext)[0]
    elif length == 127:
        ext = _ws_read_exact(sock, 8)
        if ext is None:
            return None
        length = struct.unpack("!Q", ext)[0]

    mask_key = _ws_read_exact(sock, 4) if masked else b""
    if masked and mask_key is None:
        return None

    payload = _ws_read_exact(sock, length) if length > 0 else b""
    if payload is None:
        return None

    if masked and mask_key:
        unmasked = bytes(b ^ mask_key[idx % 4] for idx, b in enumerate(payload))
        return opcode, unmasked
    return opcode, payload


class _EarthJukeboxHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: SonificationService,
    ) -> None:
        self.service = service
        self.stop_event = threading.Event()
        super().__init__(server_address, _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    server: _EarthJukeboxHTTPServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Suppress noisy per-request stderr logs during tests and real-time mouse streaming
        return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: Any) -> None:
        encoded = json.dumps(data, ensure_ascii=True).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", encoded)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object.")
        return parsed

    @staticmethod
    def _query_to_payload(query_str: str) -> dict[str, Any]:
        params = parse_qs(query_str, keep_blank_values=False)
        payload: dict[str, Any] = {}
        for key in ("x", "y", "x_ratio", "y_ratio"):
            if key in params:
                payload[key] = float(params[key][0])
        for key in ("sensor_size", "bars", "bar_index", "sample_rate"):
            if key in params:
                payload[key] = int(params[key][0])
        if "dataset" in params:
            payload["dataset"] = params["dataset"][0]
        if "touching" in params:
            payload["touching"] = params["touching"][0].lower() not in ("0", "false", "no")
        if "drums" in params:
            payload["drums"] = params["drums"][0].lower() not in ("0", "false", "no")
        if "preview_only" in params:
            payload["preview_only"] = params["preview_only"][0].lower() not in ("0", "false", "no")
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_bytes(
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                FRONTEND_DEMO_HTML.encode("utf-8"),
            )
            return

        if path == "/api/health":
            self._send_json(HTTPStatus.OK, self.server.service.current_status())
            return

        if path == "/api/datasets":
            self._send_json(
                HTTPStatus.OK,
                {"datasets": self.server.service.list_datasets()},
            )
            return

        if path.startswith("/api/datasets/") and path.endswith("/image"):
            parts = [p for p in path.split("/") if p]
            if len(parts) == 4 and parts[2].isdigit():
                try:
                    png_bytes = self.server.service.get_dataset_image_png(int(parts[2]))
                except IndexError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_bytes(HTTPStatus.OK, "image/png", png_bytes)
                return

        if path == "/api/audio.wav":
            try:
                payload = self._query_to_payload(parsed.query)
                wav_bytes, snapshot = self.server.service.render_audio_wav(payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_bytes(
                HTTPStatus.OK,
                "audio/wav",
                wav_bytes,
                extra_headers={
                    "Cache-Control": "no-store",
                    "X-EarthJukebox-Chord": str(snapshot["music"]["chord"]),
                    "X-EarthJukebox-Color": str(snapshot["analysis"]["dominant_color"]),
                    "X-EarthJukebox-Density": str(snapshot["analysis"]["density"]),
                },
            )
            return

        if path == "/api/stream":
            self._handle_sse_stream()
            return

        if path == "/ws":
            self._handle_websocket()
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            payload = self._read_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if path in ("/api/coordinates", "/api/sonify"):
            try:
                snapshot = self.server.service.process_coordinates(payload)
            except (ValueError, IndexError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, snapshot)
            return

        if path == "/api/audio":
            try:
                wav_bytes, snapshot = self.server.service.render_audio_wav(payload)
            except (ValueError, IndexError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_bytes(
                HTTPStatus.OK,
                "audio/wav",
                wav_bytes,
                extra_headers={
                    "Cache-Control": "no-store",
                    "X-EarthJukebox-Chord": str(snapshot["music"]["chord"]),
                    "X-EarthJukebox-Color": str(snapshot["analysis"]["dominant_color"]),
                    "X-EarthJukebox-Density": str(snapshot["analysis"]["density"]),
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def _handle_sse_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        sub = self.server.service.broadcast_backend.subscribe()
        try:
            hello = json.dumps(
                {"type": "connected", "status": self.server.service.current_status()}
            )
            self.wfile.write(f"event: connected\ndata: {hello}\n\n".encode("utf-8"))
            self.wfile.flush()

            while not self.server.stop_event.is_set():
                try:
                    event = sub.get(timeout=0.5)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                event_type = str(event.get("type", "message"))
                data = json.dumps(event, ensure_ascii=True)
                self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.server.service.broadcast_backend.unsubscribe(sub)

    def _handle_websocket(self) -> None:
        ws_key = self.headers.get("Sec-WebSocket-Key")
        if not ws_key:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing Sec-WebSocket-Key header."})
            return

        accept = base64.b64encode(
            hashlib.sha1((ws_key.strip() + _WS_GUID).encode("ascii")).digest()
        ).decode("ascii")

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii")
        self.connection.sendall(response)

        send_lock = threading.Lock()
        closed_event = threading.Event()
        sub = self.server.service.broadcast_backend.subscribe()

        def safe_send(opcode: int, data: bytes) -> bool:
            frame = _ws_encode_frame(opcode, data)
            with send_lock:
                try:
                    self.connection.sendall(frame)
                    return True
                except OSError:
                    closed_event.set()
                    return False

        def reader_loop() -> None:
            try:
                while not closed_event.is_set() and not self.server.stop_event.is_set():
                    decoded = _ws_read_frame(self.connection)
                    if decoded is None:
                        break
                    opcode, payload_bytes = decoded
                    if opcode == 0x8:  # Close
                        safe_send(0x8, b"")
                        break
                    if opcode == 0x9:  # Ping -> Pong
                        safe_send(0xA, payload_bytes)
                        continue
                    if opcode != 0x1:  # Ignore non-text frames
                        continue
                    try:
                        payload = json.loads(payload_bytes.decode("utf-8"))
                        self.server.service.process_coordinates(payload)
                    except (ValueError, json.JSONDecodeError, IndexError) as exc:
                        err = json.dumps({"type": "error", "error": str(exc)}).encode("utf-8")
                        safe_send(0x1, err)
            finally:
                closed_event.set()

        reader = threading.Thread(target=reader_loop, daemon=True, name="earth-jukebox-ws-reader")
        reader.start()

        try:
            hello = json.dumps(
                {"type": "connected", "status": self.server.service.current_status()}
            ).encode("utf-8")
            safe_send(0x1, hello)

            while not closed_event.is_set() and not self.server.stop_event.is_set():
                try:
                    event = sub.get(timeout=0.25)
                except queue.Empty:
                    continue
                encoded = json.dumps(event, ensure_ascii=True).encode("utf-8")
                if not safe_send(0x1, encoded):
                    break
        finally:
            closed_event.set()
            self.server.service.broadcast_backend.unsubscribe(sub)
            reader.join(timeout=0.5)


class EarthJukeboxServer:
    """Lifecycle manager for the Earth Jukebox frontend streaming server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        service: Optional[SonificationService] = None,
        enable_local_audio: bool = False,
        ambient_drums: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.service = service or SonificationService(
            enable_local_audio=enable_local_audio,
            ambient_drums=ambient_drums,
        )
        self._httpd: Optional[_EarthJukeboxHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> tuple[str, int]:
        if self._httpd is not None:
            addr = self._httpd.server_address
            return str(addr[0]), int(addr[1])
        return self.host, self.port

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    @property
    def ws_url(self) -> str:
        host, port = self.address
        return f"ws://{host}:{port}/ws"

    def start(self) -> None:
        """Start the audio sequencer and HTTP/SSE/WebSocket server threads."""
        if self._httpd is not None:
            return
        self.service.start()
        self._httpd = _EarthJukeboxHTTPServer((self.host, self.port), self.service)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="earth-jukebox-http-server",
        )
        self._thread.start()

    def serve_forever(self) -> None:
        """Start and block cleanly until interrupted."""
        self.start()
        try:
            while self._httpd is not None and not self._httpd.stop_event.is_set():
                self._httpd.stop_event.wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Shut down active streams, HTTP server, and audio engine."""
        if self._httpd is not None:
            self._httpd.stop_event.set()
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        self.service.stop()


__all__ = ["EarthJukeboxServer", "SonificationService"]
