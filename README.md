# Earth Jukebox

**Real-time tactile and computer-vision sonification of NASA Earth Information Center (EIC) visualizations** -- built for the **2026 NASA Space Apps Challenge**.

Earth Jukebox pairs a **3D-printed tactile world map** mounted over a Linux laptop trackpad (with a cross-platform mouse hover proxy for desktop exploration) with **real-time OpenCV patch analysis** and a **Daft Punk "Veridis Quo"-inspired C Natural Minor 90 BPM synthesizer**, accompanied by **spoken Text-to-Speech (TTS)** scientific interpretations for blind, low-vision, and sighted users.

---

## Highlights

- **Tactile + Pointer Exploration**: Explore planetary data physically through a 3D-printed world map overlay (`hardware/tactile-map/`) on Linux `evdev` trackpads or seamlessly with a desktop mouse cursor.
- **Real-Time Computer Vision**: Extracts a resizable sensor box (`16` to `160` px, default `40` px) and measures dominant color, intensity, edge density, spatial shape (`uniform`, `linear`, `circular`, `organic`), spottiness, and swirliness.
- **Veridis Quo-Inspired 90 BPM Sonification**: Translates visual Earth features into a locked 90 BPM, C Natural Minor (`C, D, Eb, F, G, Ab, Bb`) baroque 8-bar progression (`Cm9 -> Fm7 -> Bb9 -> Ebmaj7 -> Abmaj7 -> Fm9 -> Dm7b5 -> Gm7`) with adaptive ambient percussion.
- **Accessible by Design**: Non-blocking, interruptible TTS announces zone readings, scientific color meanings, and dataset briefings via keyboard shortcuts (`pyttsx3`, Windows `System.Speech`, or Linux `spd-say` / `espeak`).

---

## Quickstart

### 1. Install

Requires Python 3.10+ (including `tkinter` for the desktop GUI).

```bash
# Clone the repository and install the package in editable mode
pip install -e .

# Optional extras:
#   tts   -> pyttsx3 spoken output
#   midi  -> pygame MIDI playback
#   linux -> evdev trackpad driver for Linux tactile hardware
#   dev   -> pytest and ruff
pip install -e ".[tts,midi,dev]"
```

On Linux with the 3D-printed tactile map over a trackpad, also install the `linux` extra:

```bash
pip install -e ".[linux]"
```

### 2. Run

```bash
# Desktop mouse hover mode (Windows / Linux / macOS)
earth-jukebox

# Linux 3D-printed tactile map over trackpad (auto-detects touchpad)
earth-jukebox --input evdev

# Linux evdev with an explicit event device path
earth-jukebox --input evdev --device /dev/input/event7

# Equivalent module invocation
python -m earth_jukebox
```

### 3. CLI Flags & Headless Self-Test

```text
--input {mouse,evdev}   Input coordinate source (default: mouse)
--device PATH           Linux evdev device path when using --input evdev
--no-audio              Disable MIDI audio output
--no-drums              Start with ambient drums muted (toggle anytime with D)
--no-tts                Disable spoken TTS announcements
--self-test             Run headless CV, dataset, audio, and TTS diagnostics
```

Verify the full pipeline without opening a GUI window:

```bash
python -m earth_jukebox --self-test
```

---

## Controls

| Input | Action |
| :--- | :--- |
| `[Space]` | Speak the Audio Legend calibration and onboarding guide |
| `[R]` | Read aloud the current cursor zone (color, density, shape, intensity, dataset meaning) |
| `[I]` | Speak the scientific summary for the active NASA dataset |
| `[P]` | Toggle Auto-Scan Jukebox mode (pauses automatically on manual cursor movement) |
| `[D]` | Toggle the Channel 9 ambient drum groove on or off |
| `[Tab]` / `[1]`-`[8]` | Cycle to the next dataset or jump directly to dataset `1` through `8` |
| `[+]` / `[-]` or `[Scroll]` | Resize the CV sensor box (`16` to `160` px in `8` px steps; also `=`, `_`, `[`, `]`) |
| `[Esc]` | Stop audio, input, and TTS threads and exit cleanly |

---

## Repository Layout

```text
EarthJukebox/
|-- pyproject.toml                  # Package metadata, extras, and CLI entry point
|-- data/
|   `-- images/                     # 7 NASA EIC Earth science visualizations
|-- hardware/
|   `-- tactile-map/                # 3D-printable STL/3MF tactile map + reference JPG
|-- src/
|   `-- earth_jukebox/
|       |-- models.py               # TouchPoint, AnalysisResult, DatasetEntry
|       |-- cli.py                  # CLI parser, headless --self-test, app launcher
|       |-- input/                  # CoordinateInputSource, mouse proxy, Linux evdev driver
|       |-- datasets/               # DatasetManager and NASA EIC catalog.json
|       |-- vision/                 # Edge-safe sensor box extraction and CVAnalyzer
|       |-- audio/                  # C-minor music theory, MIDI backends, 90 BPM engine
|       |-- accessibility/          # Threaded interruptible AccessibilityTTS
|       `-- ui/                     # Fixed-size accessible Tkinter UI (EarthJukeboxApp)
|-- tools/
|   `-- trackpad_monitor.py         # Standalone Linux evdev trackpad coordinate viewer
|-- tests/                          # Unit and integration test suite (pytest)
`-- docs/                           # Detailed technical documentation
```

---

## Documentation

- [Architecture & Linux evdev Hardware Guide](docs/architecture.md) -- Module breakdown, end-to-end data flow diagram, cross-platform design, and step-by-step Linux `evdev` setup.
- [Sonification & Music Theory](docs/sonification.md) -- The Veridis Quo-inspired C Natural Minor 90 BPM progression, ambient percussion, and CV-to-music mappings.
- [NASA EIC Datasets & Tactile Hardware](docs/datasets.md) -- Detailed catalog of all 7 NASA visualizations in `data/images/` plus the `hardware/tactile-map/` 3D models.

---

## License

Released under the [MIT License](LICENSE).
