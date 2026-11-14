# Frontend Coordinate-to-Music Streaming API

Earth Jukebox includes a built-in, zero-extra-dependency HTTP, Server-Sent Events (SSE), RIFF WAV audio, and RFC 6455 WebSocket server (`earth_jukebox.server`). It receives image coordinates from any external or browser frontend, runs OpenCV patch analysis on the selected NASA Earth visualization, and streams quantized **75 BPM Veridis Quo C Natural Minor** music back to the frontend in real time.

---

## 1. Quickstart

Start the server from the command line:

```bash
# Start streaming server on http://127.0.0.1:8765 (audio plays in the frontend)
earth-jukebox --serve

# Custom host/port, or module invocation
python -m earth_jukebox --serve --host 0.0.0.0 --port 8765

# Optional: also play MIDI/synth locally on the host machine
earth-jukebox --serve --local-audio
```

Open `http://127.0.0.1:8765/` in any modern browser to try the built-in accessible WebAudio demo client immediately.

---

## 2. Three Ways to Send Coordinates & Receive Music

1. **Real-Time Duplex WebSocket (`ws://<host>:<port>/ws`)** -- Recommended for interactive 60 Hz pointer or trackpad UIs. The frontend sends coordinate JSON frames and receives `sonification_state` snapshots plus live `note_on`, `note_off`, and `control_change` events (`freq_hz`, `midi_note`, `note_name`, `channel`, `program`, `velocity`) synchronized to the 75 BPM clock.
2. **REST + Server-Sent Events (`POST /api/coordinates` + `GET /api/stream`)** -- Send coordinate updates via HTTP `POST` (or `POST /api/sonify`) and consume live musical events via `EventSource("/api/stream")`.
3. **On-Demand Rendered WAV Audio (`POST /api/audio` or `GET /api/audio.wav`)** -- Send image coordinates and receive ready-to-play 16-bit PCM RIFF WAV (`audio/wav`) synthesized directly for that coordinate patch.

CORS (`Access-Control-Allow-Origin: *`) is enabled on all HTTP endpoints so external frontends (React, Vite, Next.js, vanilla JS) can connect directly from another origin.

---

## 3. Coordinate Request Format

All coordinate endpoints (`POST /api/coordinates`, `POST /api/sonify`, `POST /api/audio`, WebSocket `/ws`, and query parameters on `GET /api/audio.wav`) accept either normalized or pixel coordinates:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `x_ratio`, `y_ratio` | `float` (`0.0` to `1.0`) | Required unless `x`, `y` provided | Normalized pointer position across the dataset image |
| `x`, `y` | `int` or `float` | Required unless `x_ratio`, `y_ratio` provided | Image pixel coordinates (automatically clamped to image bounds) |
| `sensor_size` | `int` (`16` to `160`) | `40` | Width and height in pixels of the analyzed square patch |
| `dataset` | `int` or `string` | Active dataset | `0`-based dataset index (from `GET /api/datasets`) or partial dataset name |
| `touching` | `bool` | `true` | Set `false` when the pointer leaves the image to silence music cleanly |
| `drums` | `bool` | Current engine state | Enable or mute Channel 9 ambient percussion |
| `bar_index` | `int` (`0` to `7`) | Current engine bar | Optional bar override inside the 8-bar progression (`Cm9`..`Gm7`) |
| `bars` | `int` (`1` to `4`) | `1` | WAV endpoints only: number of 4/4 bars to synthesize (`3.2 s` per bar at 75 BPM) |
| `sample_rate` | `int` (`8000` to `48000`) | `44100` | WAV endpoints only: output PCM sample rate in Hz |
| `preview_only` | `bool` | `false` | WAV endpoints only: when `true`, renders audio without altering the live engine |

---

## 4. HTTP & Streaming Endpoints

### `GET /api/health`
Returns server status, master BPM (`75.0`), active bar/beat/chord, subscriber count, and current dataset dimensions.

### `GET /api/datasets`
Returns all discovered NASA EIC visualizations (`index`, `name`, `category`, `unit`, `summary`, `width`, `height`, `image_url`).

### `GET /api/datasets/{index}/image`
Returns the normalized dataset image (`image/png`).

### `POST /api/coordinates` (alias `POST /api/sonify`)
Analyzes the patch around the received coordinates, updates `CMinorAudioEngine`, broadcasts a `sonification_state` event to connected `/api/stream` and `/ws` clients, and returns the immediate JSON snapshot:

```json
{
  "type": "sonification_state",
  "timestamp_ms": 1731614526123,
  "dataset": {
    "index": 0,
    "name": "Aerosol Optical Depth and PM2.5",
    "category": "Atmospheric Composition & Air Quality",
    "unit": "AOD / ug m^-3",
    "width": 960,
    "height": 540,
    "meaning": "..."
  },
  "coordinates": {
    "x": 384,
    "y": 216,
    "x_ratio": 0.4,
    "y_ratio": 0.4,
    "sensor_size": 48,
    "sensor_bbox": [360, 192, 408, 240],
    "touching": true
  },
  "analysis": {
    "dominant_color": "Orange",
    "intensity": 0.7421,
    "density": 0.6182,
    "shape": "circular",
    "shape_score": 0.812,
    "spottiness": 0.34,
    "swirliness": 0.49
  },
  "description": "Dataset: Aerosol Optical Depth and PM2.5. Position X 384, Y 216...",
  "music": {
    "bpm": 75.0,
    "bar_index": 0,
    "beat": 2,
    "chord": "Cm9",
    "touching": true,
    "drums_enabled": true,
    "expression_ccs": {
      "reverb": 74,
      "chorus": 70,
      "modulation": 35,
      "brightness": 103,
      "expression": 108,
      "pitch_bend": 8192
    },
    "chord_voicing": { "channel": 0, "velocity": 68, "notes": [] },
    "bass_note": { "channel": 6, "notes": [{ "midi_note": 36, "note_name": "C2", "freq_hz": 65.4064, "velocity": 72 }] },
    "baroque_arp": { "channel": 8, "midi_note": 79, "note_name": "G5", "freq_hz": 783.9909, "velocity": 89, "gate_steps": 2 },
    "spottiness_accent": null,
    "primary_voice": {
      "channel": 3,
      "midi_note": 75,
      "note_name": "Eb5",
      "freq_hz": 622.254,
      "harmony_midi_note": 72,
      "harmony_note_name": "C5",
      "harmony_freq_hz": 523.2511,
      "velocity": 92,
      "volume": 0.65,
      "motif": "Eighth Walk",
      "duration_sec": 0.4
    }
  }
}
```

### `GET /api/stream` (Server-Sent Events)
Streams real-time events formatted as `event: <type>\ndata: <json>\n\n`:

- `connected`: Initial connection confirmation + current server status.
- `sonification_state`: Published whenever new coordinates update the analyzed zone.
- `note_on` / `note_off`: Live quantized MIDI note triggers including `channel`, `program` (General MIDI instrument), `midi_note`, `note_name`, `freq_hz`, and `velocity`.
- `control_change`: Spatial MIDI CC updates (`91` reverb, `93` chorus, `1` modulation, `74` brightness, `11` expression).

### `GET /ws` (WebSocket)
Accepts JSON coordinate messages in the same schema as `POST /api/coordinates` and pushes `connected`, `sonification_state`, `note_on`, `note_off`, and `control_change` JSON frames directly over the WebSocket.

### `POST /api/audio` and `GET /api/audio.wav`
Returns synthesized `audio/wav` audio for the supplied coordinates along with response headers `X-EarthJukebox-Chord`, `X-EarthJukebox-Color`, and `X-EarthJukebox-Density`.

---

## 5. Client Integration Examples

### Browser WebSocket + WebAudio

```javascript
const ws = new WebSocket("ws://127.0.0.1:8765/ws");
const audioCtx = new AudioContext();

ws.onopen = () => {
  ws.send(JSON.stringify({ x_ratio: 0.45, y_ratio: 0.52, sensor_size: 48, dataset: 0 }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "note_on" && msg.channel !== 9) {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = msg.channel === 8 ? "square" : msg.channel === 6 ? "sawtooth" : "sine";
    osc.frequency.value = msg.freq_hz;
    gain.gain.value = (msg.velocity / 127) * 0.12;
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  }
};
```

### Fetching Synthesized WAV Audio with `curl`

```bash
# Render 1 bar of Veridis Quo C-minor music for coordinates (480, 270)
curl -X POST http://127.0.0.1:8765/api/audio ^
  -H "Content-Type: application/json" ^
  -d "{\"x\": 480, \"y\": 270, \"sensor_size\": 64, \"bars\": 1}" ^
  --output zone_music.wav
```
