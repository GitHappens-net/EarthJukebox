"""Built-in accessible WebAudio reference frontend served at ``GET /``."""

from __future__ import annotations

FRONTEND_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Earth Jukebox - Frontend Music Stream</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      background: #0b1120; color: #e2e8f0; display: flex; flex-direction: column;
      align-items: center; padding: 16px; gap: 12px;
    }
    .panel {
      width: 960px; max-width: 100%; background: #1e293b; border: 1px solid #334155;
      border-radius: 8px; padding: 12px 16px; box-sizing: border-box;
    }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, select, input[type="range"] {
      background: #0f172a; color: #f8fafc; border: 1px solid #475569;
      border-radius: 6px; padding: 6px 10px; font-size: 14px;
    }
    button:hover { border-color: #38bdf8; cursor: pointer; }
    #mapWrap { position: relative; width: 960px; max-width: 100%; }
    #mapCanvas {
      width: 100%; height: auto; display: block; border-radius: 8px;
      border: 2px solid #334155; cursor: crosshair; background: #020617;
    }
    .sr-box {
      width: 100%; box-sizing: border-box; background: #0f172a; color: #7dd3fc;
      border: 1px solid #334155; border-radius: 6px; padding: 8px; font-size: 14px;
    }
    code { color: #38bdf8; }
  </style>
</head>
<body>
  <div class="panel">
    <div class="row" style="justify-content: space-between;">
      <strong>Earth Jukebox - Coordinate to Music Stream</strong>
      <div class="row">
        <button id="startBtn" type="button">Start WebAudio Stream</button>
        <button id="wavBtn" type="button">Play Rendered WAV Clip</button>
        <label for="datasetSelect">Dataset:</label>
        <select id="datasetSelect" aria-label="NASA Dataset"></select>
        <label for="boxRange">Box (<span id="boxLabel">40</span>px):</label>
        <input id="boxRange" type="range" min="16" max="160" step="8" value="40" />
      </div>
    </div>
  </div>

  <div id="mapWrap">
    <canvas id="mapCanvas" width="960" height="540" tabindex="0"
      aria-label="Interactive NASA Earth visualization canvas. Move pointer to stream music."></canvas>
  </div>

  <div class="panel">
    <div class="row" style="justify-content: space-between; margin-bottom: 8px;">
      <span id="musicStatus">Music: Stopped (click Start WebAudio Stream, then hover map)</span>
      <span id="coordStatus">Coordinates: --</span>
    </div>
    <label for="srOutput" style="display:block; font-size:12px; color:#94a3b8; margin-bottom:4px;">
      Screen Reader Live Description:
    </label>
    <input id="srOutput" class="sr-box" type="text" readonly aria-live="polite"
      value="Connect WebAudio and hover over the image to receive live C-minor music." />
  </div>

  <script>
    const canvas = document.getElementById("mapCanvas");
    const ctx = canvas.getContext("2d");
    const datasetSelect = document.getElementById("datasetSelect");
    const boxRange = document.getElementById("boxRange");
    const boxLabel = document.getElementById("boxLabel");
    const startBtn = document.getElementById("startBtn");
    const wavBtn = document.getElementById("wavBtn");
    const musicStatus = document.getElementById("musicStatus");
    const coordStatus = document.getElementById("coordStatus");
    const srOutput = document.getElementById("srOutput");

    let datasets = [];
    let currentImage = new Image();
    let audioCtx = null;
    let ws = null;
    let activeVoices = new Map();
    let lastCoords = { x_ratio: 0.5, y_ratio: 0.5 };
    let lastSentMs = 0;

    function ensureAudioContext() {
      if (!audioCtx) {
        const AudioContextCls = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContextCls();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume();
      }
    }

    function playNoteEvent(evt) {
      if (!audioCtx) return;
      const key = `${evt.channel}:${evt.midi_note}`;
      if (evt.type === "note_off") {
        const voice = activeVoices.get(key);
        if (voice) {
          const now = audioCtx.currentTime;
          voice.gain.gain.cancelScheduledValues(now);
          voice.gain.gain.setTargetAtTime(0.0001, now, 0.04);
          voice.osc.stop(now + 0.16);
          activeVoices.delete(key);
        }
        return;
      }
      if (evt.type !== "note_on") return;

      const old = activeVoices.get(key);
      if (old) {
        try { old.osc.stop(); } catch (_) {}
        activeVoices.delete(key);
      }

      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      if (evt.channel === 9) {
        osc.type = "triangle";
        osc.frequency.setValueAtTime(evt.midi_note === 36 ? 72 : 220, now);
      } else if (evt.channel === 8) {
        osc.type = "square";
        osc.frequency.setValueAtTime(evt.freq_hz, now);
      } else if (evt.channel === 6) {
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(evt.freq_hz, now);
      } else {
        osc.type = "sine";
        osc.frequency.setValueAtTime(evt.freq_hz, now);
      }

      const amp = Math.max(0.01, Math.min(0.18, (evt.velocity / 127) * 0.12));
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(amp, now + 0.015);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      activeVoices.set(key, { osc, gain });
      if (evt.channel === 9) {
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
        osc.stop(now + 0.10);
        activeVoices.delete(key);
      }
    }

    function connectWebSocket() {
      ensureAudioContext();
      if (ws && ws.readyState === WebSocket.OPEN) return;
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${location.host}/ws`);
      ws.onopen = () => {
        musicStatus.textContent = "Music: WebSocket Connected (75 BPM Veridis Quo C-Minor)";
        sendCoordinates(lastCoords.x_ratio, lastCoords.y_ratio, true);
      };
      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data);
        if (data.type === "sonification_state") {
          srOutput.value = data.description;
          musicStatus.textContent =
            `Chord: ${data.music.chord} | Color: ${data.analysis.dominant_color} | ` +
            `Note: ${data.music.primary_voice ? data.music.primary_voice.note_name : "--"}`;
        } else if (data.type === "note_on" || data.type === "note_off") {
          playNoteEvent(data);
        }
      };
      ws.onclose = () => {
        musicStatus.textContent = "Music: Disconnected";
      };
    }

    function sendCoordinates(xRatio, yRatio, force = false) {
      const now = performance.now();
      if (!force && now - lastSentMs < 40) return;
      lastSentMs = now;
      lastCoords = { x_ratio: xRatio, y_ratio: yRatio };
      coordStatus.textContent = `Coordinates: (${xRatio.toFixed(3)}, ${yRatio.toFixed(3)})`;
      const payload = {
        x_ratio: xRatio,
        y_ratio: yRatio,
        sensor_size: Number(boxRange.value),
        dataset: Number(datasetSelect.value || 0),
        touching: true,
      };
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      } else {
        fetch("/api/coordinates", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).then((r) => r.json()).then((data) => {
          if (data.description) srOutput.value = data.description;
          if (data.music) {
            musicStatus.textContent =
              `Chord: ${data.music.chord} | Color: ${data.analysis.dominant_color} | ` +
              `Note: ${data.music.primary_voice ? data.music.primary_voice.note_name : "--"}`;
          }
        }).catch(() => {});
      }
    }

    function drawCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (currentImage.complete && currentImage.naturalWidth > 0) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
      }
    }

    async function loadDatasets() {
      const res = await fetch("/api/datasets");
      const data = await res.json();
      datasets = data.datasets || [];
      datasetSelect.innerHTML = "";
      datasets.forEach((ds) => {
        const opt = document.createElement("option");
        opt.value = String(ds.index);
        opt.textContent = `${ds.index + 1}. ${ds.name}`;
        datasetSelect.appendChild(opt);
      });
      if (datasets.length > 0) {
        currentImage.onload = drawCanvas;
        currentImage.src = datasets[0].image_url;
      }
    }

    datasetSelect.addEventListener("change", () => {
      const idx = Number(datasetSelect.value);
      const ds = datasets[idx];
      if (ds) {
        currentImage.onload = drawCanvas;
        currentImage.src = ds.image_url;
        sendCoordinates(lastCoords.x_ratio, lastCoords.y_ratio, true);
      }
    });

    boxRange.addEventListener("input", () => {
      boxLabel.textContent = boxRange.value;
      sendCoordinates(lastCoords.x_ratio, lastCoords.y_ratio, true);
    });

    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const xr = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const yr = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      sendCoordinates(xr, yr, false);
    });

    canvas.addEventListener("mouseleave", () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ touching: false }));
      }
    });

    startBtn.addEventListener("click", connectWebSocket);

    wavBtn.addEventListener("click", async () => {
      ensureAudioContext();
      const url = `/api/audio.wav?x_ratio=${lastCoords.x_ratio.toFixed(4)}` +
        `&y_ratio=${lastCoords.y_ratio.toFixed(4)}&sensor_size=${boxRange.value}` +
        `&dataset=${Number(datasetSelect.value || 0)}&bars=1`;
      const audio = new Audio(url);
      await audio.play();
    });

    loadDatasets();
  </script>
</body>
</html>
"""
