# Earth Jukebox (TerraGroove)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![NASA Space Apps](https://img.shields.io/badge/NASA%20Space%20Apps-2026-e2a95c.svg)](https://www.spaceappschallenge.org/)

**Real-time tactile and computer-vision sonification of NASA Earth Information Center (EIC) visualizations** — built for the **2026 NASA Space Apps Challenge**.

Earth Jukebox (**TerraGroove**) pairs a **3D-printed tactile world map** mounted over a laptop trackpad (with automatic browser/mouse pointer fallback on Windows, macOS, and Linux) with **real-time OpenCV patch analysis** and a **Daft Punk *"Veridis Quo"*-inspired C Natural Minor 75 BPM synthesizer**, turning planetary data into an interactive, accessible musical landscape.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture & Repository Layout](#architecture--repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
  - [1. TerraGroove React + pywebview App (Recommended)](#1-terragroove-react--pywebview-app-recommended)
  - [2. Standalone CLI & Tkinter Desktop App](#2-standalone-cli--tkinter-desktop-app)
- [Controls & Keyboard Shortcuts](#controls--keyboard-shortcuts)
- [Testing & Verification](#testing--verification)
- [Documentation](#documentation)
- [License](#license)

---

## Key Features

- **Cross-Platform Touchpad & Pointer Exploration**: Explore planetary datasets physically through a 3D-printed world map overlay (`hardware/tactile-map/`) using raw Linux `evdev` events, or seamlessly via browser/mouse pointer events on Windows, macOS, and Linux (`--input auto`).
- **Real-Time Computer Vision Pipeline**: Extracts an edge-safe sensor patch (`16` to `160` px, default `40` px) and computes dominant color, luminance/saturation intensity, Canny edge density, spatial morphology (`uniform`, `linear`, `circular`, `organic`), spottiness, and rotational swirliness.
- **Veridis Quo-Inspired 75 BPM Sonification**: Translates visual Earth features into a locked 75 BPM, C Natural Minor (`C, D, Eb, F, G, Ab, Bb`) baroque 8-bar harmonic progression (`Cm9 -> Fm7 -> Bb9 -> Ebmaj7 -> Abmaj7 -> Fm9 -> Dm7b5 -> Gm7`) with multi-voice synthesis, adaptive percussion, and cross-platform MIDI/waveform backends.
- **Modern React UI + Native Python Bridge**: Built with React 19, TypeScript, Vite, and Tailwind CSS (`frontend/`), bridged directly to the Python sonification and computer-vision service (`backend/`) through `pywebview`.

---

## Architecture & Repository Layout

```text
EarthJukebox/
├── pyproject.toml                  # Python package metadata, extras, and CLI entry point
├── backend/
│   ├── main.py                     # pywebview desktop launcher & JS-to-Python API bridge
│   ├── earth_jukebox/              # Sonification service, CV analyzer, datasets, and synth engine
│   └── tests/                      # Backend unit and service tests
├── frontend/
│   ├── package.json                # React 19, TypeScript, Vite, and Tailwind CSS configuration
│   └── src/                        # TerraGroove UI components, trackpad hooks, and pywebview client
├── src/
│   └── earth_jukebox/              # Standalone CLI, Tkinter desktop UI, CV, audio, and input drivers
├── data/
│   └── images/                     # 7 curated NASA EIC Earth science visualizations
├── hardware/
│   └── tactile-map/                # 3D-printable STL/3MF tactile world map + reference projection
├── tools/
│   └── trackpad_monitor.py         # Standalone Linux evdev trackpad coordinate diagnostic tool
├── tests/                          # Core unit and integration test suite (pytest)
└── docs/                           # Technical documentation (architecture, sonification, datasets)
```

---

## Prerequisites

- **Python**: `3.10` or newer (`tkinter` required only for the standalone Tkinter UI)
- **Node.js & npm**: Node.js `18+` and `npm` (for building or running the React frontend)
- **Linux Trackpad Hardware Mode (Optional)**:
  - Read access to `/dev/input/event*` (add your user to the `input` group: `sudo usermod -aG input "$USER"`)
  - On Fedora/Linux, `pywebview` prefers `PySide6` WebEngine when installed (`PySide6-Addons` / `QtWebEngineCore`) to avoid system Qt conflicts.

---

## Installation

### 1. Python Backend & Core Package

From the repository root, install the package and optional runtime extras:

```bash
# Install core package (OpenCV, NumPy, Pillow)
pip install -e .

# Install audio, desktop webview, and development dependencies
pip install -e ".[midi,dev]" pywebview
```

Optional platform extras:

```bash
# Linux raw evdev trackpad driver
pip install -e ".[linux]"

# Spoken Text-to-Speech (TTS) announcements for standalone mode
pip install -e ".[tts]"
```

### 2. React Frontend (`frontend/`)

Install the Node.js dependencies and build the production bundle for `pywebview`:

```bash
npm --prefix frontend install
npm --prefix frontend run build
```

---

## Running the Application

### 1. TerraGroove React + pywebview App (Recommended)

Once `frontend/dist` is built (`npm --prefix frontend run build`), launch the integrated application:

```bash
# Automatic input selection (evdev on Linux, browser pointer events on Windows/macOS)
python backend/main.py --input auto
```

You can also select the input backend explicitly:

```bash
# Portable mouse / touchpad pointer mode (Windows, macOS, Linux)
python backend/main.py --input browser

# Raw Linux evdev touchpad mode (auto-detects touchpad or accepts --device)
python backend/main.py --input evdev
python backend/main.py --input evdev --device /dev/input/event7
```

#### Frontend Hot-Reload Development Mode

To iterate on the React UI with Vite hot module replacement (HMR):

1. Start the Vite dev server on `http://localhost:5173`:
   ```bash
   npm --prefix frontend run dev
   ```
2. In a second terminal, launch the backend pointing to the dev server:
   ```bash
   python backend/main.py --input browser --dev
   ```

---

### 2. Standalone CLI & Tkinter Desktop App

You can also run the standalone Python package directly without `pywebview`:

```bash
# Desktop mouse hover mode (Windows / Linux / macOS)
earth-jukebox

# Linux 3D-printed tactile map over trackpad
earth-jukebox --input evdev --device /dev/input/event7

# Equivalent module invocation
python -m earth_jukebox
```

#### CLI Flags

```text
--input {mouse,evdev}   Input coordinate source (default: mouse)
--device PATH           Linux evdev device path when using --input evdev
--no-audio              Disable MIDI/synth audio output
--no-drums              Start with ambient drums muted (toggle anytime with D)
--self-test             Run headless CV, dataset, and audio diagnostics
```

---

## Controls & Keyboard Shortcuts

### TerraGroove Web UI (`backend/main.py`)

| Input | Action |
| :--- | :--- |
| **Pointer / Trackpad** | Press/drag (browser mode) or touch (Linux `evdev` mode) across the map to sonify in real time |
| `[1]` – `[7]` | Jump directly to NASA dataset `1` through `7` |
| `[ArrowUp]` / `[ArrowLeft]` | Select previous dataset |
| `[ArrowDown]` / `[ArrowRight]` | Select next dataset |
| `[?]` | Open the Quick Start Instructions view |
| `[Esc]` | Return to the Explore view |

### Standalone Tkinter UI (`earth-jukebox`)

| Input | Action |
| :--- | :--- |
| `[Space]` | Speak the Audio Legend calibration and onboarding guide |
| `[R]` | Read aloud the current cursor zone (color, density, shape, intensity, dataset meaning) |
| `[I]` | Speak the scientific summary for the active NASA dataset |
| `[P]` | Toggle Auto-Scan Jukebox mode (pauses automatically on manual cursor movement) |
| `[D]` | Toggle the Channel 9 ambient drum groove on or off |
| `[Tab]` / `[1]`–`[8]` | Cycle to the next dataset or jump directly to dataset `1` through `8` |
| `[+]` / `[-]` or `[Scroll]` | Resize the CV sensor box (`16` to `160` px in `8` px steps; also `=`, `_`, `[`, `]`) |
| `[Esc]` | Stop audio, input, and TTS threads and exit cleanly |

---

## Testing & Verification

Run the headless self-test to verify dataset discovery, OpenCV sensor box extraction, and C-Minor pitch quantization without opening a GUI window:

```bash
python -m earth_jukebox --self-test
```

Run the automated test suite and linter:

```bash
# Run core and backend test suites
pytest tests backend/tests

# Lint with Ruff
ruff check .
```

---

## Documentation

- [Architecture & Linux `evdev` Hardware Guide](docs/architecture.md) — Module breakdown, end-to-end data flow diagram, cross-platform design, and step-by-step Linux `evdev` setup.
- [Sonification & Music Theory](docs/sonification.md) — The *Veridis Quo*-inspired C Natural Minor 75 BPM progression, ambient percussion, and CV-to-music mappings.
- [NASA EIC Datasets & Tactile Hardware](docs/datasets.md) — Detailed catalog of all 7 NASA visualizations in `data/images/` plus the `hardware/tactile-map/` 3D models.

---

## License

Released under the [MIT License](LICENSE).
