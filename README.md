# Earth Jukebox

**Earth Jukebox** is a real-time, accessible visual-to-audio sonification system built for the **2026 NASA Space Apps Challenge** ("Earth Jukebox" / NASA Earth Information Center). It translates NASA Earth Information Center (EIC) visualizations into a quantized, beat-synchronized C-minor musical soundscape paired with spoken audio descriptions for blind, low-vision, and sighted users alike.

---

## Challenge Overview & Vision

NASA's Earth Information Center produces rich visualizations tracking global climate, aerosols, wildfire risk, ocean circulation, and greenhouse gases. Presenting Earth science through sight alone limits who can access it. **Earth Jukebox** bridges that gap through multi-sensory sonification:

- **Tactile & Pointer Exploration**: Designed for a 3D-printed tactile world map mounted over a Linux laptop trackpad (`evdev`), with a full-featured Windows mouse hover proxy for development and desktop use.
- **Data-Driven Musicality**: Balances accurate computer-vision feature extraction with strict music-theory rulesets (locked 90 BPM 16th-note clock, C Natural Minor scale quantization, 4-bar ambient harmonic chord progression, and adaptive Channel 9 ambient percussion) so complex planetary data remains engaging rather than noisy.
- **Screen-Free Accessibility**: Keyboard-first navigation and asynchronous Text-to-Speech (TTS) allow full operation without relying on visual cues.

---

## Repository Structure

```text
EarthJukebox/
|-- 3D files/
|   |-- World Map.3mf               # 3D printable tactile world map model
|   |-- World Map.stl               # STL mesh for tactile trackpad overlay
|   `-- World Map.jpg               # Reference visual map aligned to the 3D model
|-- images/
|   |-- Aerosol Optical Depth.png
|   |-- Aerosols_2025-08-19T20_00_00_1920x1080.png
|   |-- ecco2_and_grid.jpg
|   |-- FireWeatherIndex_CONUS_080126_1920x1080.png
|   |-- FWI_forecast_latest.png
|   |-- global_co2_airs_60South_720x480.00253.png
|   `-- Screenshot 2026-11-14 161145.png
|-- backend/
|   |-- earth_jukebox.py            # Main 4-module sonification app & CLI
|   `-- tests/
|       |-- trackpad.py             # Linux evdev raw touchpad monitor/reader
|       `-- test_earth_jukebox.py   # Automated pytest verification suite
|-- LICENSE
`-- README.md
```

---

## Modular Architecture

The application in `backend/earth_jukebox.py` is structured into four cleanly separated object-oriented modules sharing immutable dataclasses:

### 1. Input Proxy & NASA EIC Dataset Manager
- **`TouchPoint` & `CoordinateInputSource`**: Shares the exact `TouchPoint(x, y, x_ratio, y_ratio, touching)` dataclass and bounded `queue.Queue(maxsize=2)` contract used by `backend/tests/trackpad.py`.
- **`WindowsMouseInputProxy`**: Tracks mouse hover coordinates on the Tkinter canvas (`<Motion>`, `<Enter>`, `<Leave>`, `<ButtonPress-1>`, `<B1-Motion>`) on Windows.
- **`EvdevTrackpadInputProxy`**: Wraps `find_touchpad` and `read_touchpad` directly from `backend/tests/trackpad.py` so Linux hardware trackpad events drop straight into the CV and Audio modules without changing downstream logic.
- **`extract_sensor_box`**: Extracts an edge-safe `40x40` pixel bounding box around the active cursor or touch point, clamping and shifting at corners/edges (`(0, 0)`, `(width, height)`, or negative coordinates) so it never crashes or slices empty regions.
- **`DatasetManager`**: Discovers NASA imagery in `images/` (plus `3D files/World Map.jpg`), normalizes oversized frames (such as the `6840x3420` ECCO2 frame) to `<= 960x540` with `cv2.INTER_AREA`, caches decoded frames in memory, and attaches curated scientific metadata and color-scale meanings from `NASA_EIC_CATALOG`.

### 2. Computer Vision Analysis (`CVAnalyzer`)
Continuously analyzes the `40x40` sensor patch in real time:
- **Dominant Color (`Black`, `White`, `Red`, `Orange`, `Green`, `Blue`)**:
  - Converts BGR patches to HSV and partitions pixels across the six categories.
  - Captures low-luminance, high-saturation deep crimson pixels (such as "Extreme" Fire Weather Index zones) as `Red`.
  - Maps violet mineral dust plumes (GEOS aerosols) to `Orange`.
  - Applies a `0.18` chromatic salience threshold so glowing streamlines over dark ocean or night-side backgrounds (ECCO2 currents, GEOS aerosols) are not drowned out by surrounding dark pixels.
- **Intensity (`0.0` to `1.0`)**: Weighted combination of mean HSV Value (`0.65`) and Saturation (`0.35`).
- **Density (`0.0` to `1.0`)**: Applies a `3x3` Gaussian blur and `cv2.Canny(50, 150)` edge detector, normalizing edge pixel density against `0.22` to distinguish calm oceans (`< 0.30`), coastlines/plumes (`0.30 - 0.70`), and turbulent storms or fragmented wildfire clusters (`> 0.70`).

### 3. Beat-Synchronized C-Minor Audio Engine (`CMinorAudioEngine`)
Runs on a dedicated daemon thread driven by a **locked 90 BPM 16th-note master clock** (`0.6667 s` per beat, `0.1667 s` per 16th step, 16 steps per 4/4 bar). Every pitch emitted across melody, chord pads, and bass is strictly quantized to **C Natural Minor (`C, D, Eb, F, G, Ab, Bb`)**.

#### Multi-Layered Arrangement
1. **4-Bar C-Minor Harmonic Bed (`Channel 0` Warm Pad & `Channel 6` Electric Finger Bass)**:
   - Cycles through `Cm9 -> Abmaj7 -> Fm7 -> Gm7` on bar downbeats with spatial MIDI Reverb (`CC 91`) and Chorus (`CC 93`).
2. **Adaptive Ambient Percussion (`Channel 9` General MIDI Drums)**:
   - Plays a subtle 90 BPM atmospheric groove combining heartbeat kick (`36`), ambient side-stick backbeat on Beats 2 and 4 (`37`), shaker (`70`), closed/open hi-hats (`42`/`46`), and high conga accents (`63`) scaled dynamically by visual density and intensity. Toggleable anytime with **`D`**.
3. **Beat-Quantized Data Sonification Voice (`Channels 1-5, 7`)**:
   - Triggers strictly on the 90 BPM beat grid (`should_trigger_data_step`) with quarter-beat alignment when entering new zones and C-minor harmonic dyads on downbeats.

#### Sonification Mapping Rules

| Visual Metric | Value / Category | Musical Mapping |
| :--- | :--- | :--- |
| **Dominant Color** | `Black` | Ch 1 Warm Pad (`C2, G2, C3, Eb3`) |
| **Dominant Color** | `Blue` | Ch 2 New Age Pad (`C3, Eb3, G3, C4`) |
| **Dominant Color** | `Green` | Ch 3 Marimba (`C3, F3, G3, Bb3, Eb4`) |
| **Dominant Color** | `White` | Ch 4 Vibraphone (`C4, G4, Bb4, C5, Eb5`) |
| **Dominant Color** | `Orange` | Ch 5 Electric Piano 1 (`D4, Eb4, Ab4, Bb4, D5`) |
| **Dominant Color** | `Red` | Ch 7 Lead 2 Sawtooth (`D5, Eb5, G5, Ab5, Bb5, C6`) |
| **Edge Density** | `0.00 - 0.12` | Whole Note (triggers on Beat 1 / Step `0`), quarter-note shaker |
| **Edge Density** | `0.12 - 0.30` | Half Note (triggers on Beats 1 & 3 / Steps `0, 8`) |
| **Edge Density** | `0.30 - 0.70` | 8th-Note Melodic Walk (even 16th steps), 8th hi-hat groove |
| **Edge Density** | `0.70 - 1.00` | 16th-Note Arpeggio (every 16th step), syncopated percussion |
| **Intensity** | `0.00 - 1.00` | MIDI velocity (`30 - 127`) & normalized amplitude (`0.15 - 0.95`) |

Audio backends are selected automatically: `pygame.midi` if installed, native Windows `winmm.dll` (`WinmmMidiBackend`) with zero extra audio dependencies on Windows, or `NullAudioBackend` in `--no-audio` / headless modes.

### 4. Accessibility, Keyboard Controls & Threaded TTS (`AccessibilityTTS`)
Speaks announcements asynchronously on a dedicated interruptible worker thread using `pyttsx3` when installed, with automatic fallback to Windows `System.Speech` or Linux `spd-say` / `espeak`.

| Key | Action |
| :--- | :--- |
| `Space` | Speaks the Audio Legend calibration and onboarding guide |
| `R` | Reads aloud the current cursor zone (color, density tier, intensity %, and dataset-specific Earth science interpretation) |
| `I` | Speaks the full scientific briefing for the active NASA dataset |
| `P` | Toggles Auto-Scan Jukebox mode (glides across the visualization on tempo; automatically pauses on manual mouse movement) |
| `D` | Toggles the Channel 9 ambient drum groove on or off |
| `Tab` | Cycles to the next NASA dataset and announces its name |
| `1` - `8` | Jumps directly to dataset 1 through 8 and announces its name |
| `Esc` | Cleanly stops audio, input, and TTS threads and closes the application |

---

## Included NASA EIC Datasets (`images/` & `3D files/`)

1. **Global PM2.5 Aerosol Optical Depth (2019)** (`images/Aerosol Optical Depth.png`) - Fine particulate matter concentrations (`0 - 139 ug/m3`) from MODIS, MISR, and SeaWiFS.
2. **GEOS Global Aerosol Transport (Aug 2025)** (`images/Aerosols_2025-08-19T20_00_00_1920x1080.png`) - Sea salt, Saharan mineral dust plumes, and wildfire smoke transport.
3. **ECCO2 Perpetual Ocean Currents & Temperature** (`images/ecco2_and_grid.jpg`) - High-resolution surface ocean circulation currents from cold polar waters to warm equatorial streams.
4. **North America Fire Weather Index (Aug 2026)** (`images/FireWeatherIndex_CONUS_080126_1920x1080.png`) - Regional wildfire weather danger index from Very Low to Extreme.
5. **Global Fire Weather Index Forecast (Nov 2026)** (`images/FWI_forecast_latest.png`) - Global Fire Weather Index forecast across continents.
6. **AIRS Global Tropospheric CO2 (365-435 ppm)** (`images/global_co2_airs_60South_720x480.00253.png`) - Mid-tropospheric carbon dioxide distribution from NASA Aqua/AIRS.
7. **NASA SVS 3D Carbon Dioxide Column (May 2021)** (`images/Screenshot 2026-11-14 161145.png`) - Global 3D carbon dioxide column visualization (`385 - 425 ppm`, global mean `416.8 ppm`).
8. **World Topography and Bathymetry** (`3D files/World Map.jpg`) - Reference map matching `3D files/World Map.stl` and `3D files/World Map.3mf`.

---

## Installation & Usage

### Prerequisites
- Python 3.10+ (verified on Python 3.13)
- Required packages: `opencv-python`, `numpy`, `pillow` (plus `tkinter`, included with standard Python on Windows)
- Optional packages: `pyttsx3` (otherwise uses built-in Windows `System.Speech` or Linux `spd-say`/`espeak`), `pygame` (otherwise uses built-in Windows `winmm.dll` MIDI synthesizer), `evdev` (Linux trackpad mode), `pytest` (tests)

### Running on Windows (Mouse Hover Proxy)
```powershell
python backend/earth_jukebox.py
```

### Running on Linux (3D-Printed Tactile Map over Trackpad)
Ensure the user has read permission on `/dev/input/event*` (for example, membership in the `input` group) and `python-evdev` installed:
```bash
python3 backend/earth_jukebox.py --input evdev
# Or specify the touchpad event device explicitly:
python3 backend/earth_jukebox.py --input evdev --device /dev/input/event7
```

You can also test raw Linux trackpad coordinates standalone with:
```bash
python3 backend/tests/trackpad.py
```

### CLI Options
```text
--input {mouse,evdev}   Input coordinate source (default: mouse)
--device PATH           Linux evdev device path when using --input evdev
--no-audio              Disable MIDI audio output
--no-drums              Start with ambient drums muted (toggle anytime with D)
--no-tts                Disable spoken TTS announcements
--self-test             Run headless CV, dataset, audio, and TTS diagnostics
```

---

## Verification & Testing

Run the automated pytest suite and the headless diagnostic check:
```powershell
pytest backend/tests/test_earth_jukebox.py -v
python backend/earth_jukebox.py --self-test
```
