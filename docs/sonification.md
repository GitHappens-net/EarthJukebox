# Sonification & Music Theory Design

Earth Jukebox uses a structured, rule-driven musical engine (`CMinorAudioEngine`) heavily inspired by Daft Punk's *"Veridis Quo"*. Rather than mapping raw pixel values to unquantized sine waves (which quickly become fatiguing during prolonged exploration), every note, chord, arpeggio, and drum hit locks to a cohesive harmonic and rhythmic grid.

---

## 1. Master Clock & C Natural Minor Quantization

- **Tempo**: Constant **90 BPM** (`BPM = 90.0`).
- **Beat Grid**:
  - Quarter beat: `0.6667 s` (`60.0 / 90.0`)
  - 16th-note step: `0.1667 s` (`BEAT_DURATION_SEC / 4.0`), 16 steps per 4/4 bar
- **Scale Quantization**: Every MIDI pitch across pads, bass, arpeggiator, and data melody voices is strictly quantized to **C Natural Minor**:
  - Pitch classes: `C, D, Eb, F, G, Ab, Bb` (`{0, 2, 3, 5, 7, 8, 10}` mod 12).
  - Non-scale pitches are snapped to the nearest valid C Natural Minor scale degree (`snap_to_c_minor`), eliminating dissonant out-of-key clashes regardless of dataset or cursor position.

---

## 2. Veridis Quo-Inspired 8-Bar Harmonic Architecture

The harmonic engine cycles through `VERIDIS_QUO_PROGRESSION`, an 8-bar baroque descending circle-of-fifths progression in C Natural Minor, keeping all chord tones strictly inside `{C, D, Eb, F, G, Ab, Bb}`:

```text
Bar 1: Cm9    -> Bar 2: Fm7    -> Bar 3: Bb9    -> Bar 4: Ebmaj7
Bar 5: Abmaj7 -> Bar 6: Fm9    -> Bar 7: Dm7b5  -> Bar 8: Gm7
```

Bar 7 uses `Dm7b5` (`D - F - Ab - C`) and Bar 8 resolves to `Gm7` (`G - Bb - D - F`), avoiding `B` natural so 100% of harmonic voicings remain strictly within C Natural Minor.

### Multi-Layered Arrangement

1. **Vintage Organ & Warm Pad Bed (`Channel 0` Drawbar Organ, `Channel 10` Warm Pad, `Channel 12` Church Organ)**: Holds dreamy 7th and 9th harmonic voicings on bar boundaries, enveloped with spatial MIDI Reverb (`CC 91`) and Chorus (`CC 93`).
2. **French-Touch Synth Bass (`Channel 6` Synth Bass 1)**: Plays warm root/octave pulses locked to the 8-bar chord changes.
3. **Baroque Arpeggiator & Data Lead Voices (`Channel 8` Square Lead Arp, `Channels 1-5, 7` Color Voices, `Channel 11` Spottiness Marimba)**: Cascading melodic patterns, harmonic turns, and staccato counter-accents driven in real time by the cursor's `AnalysisResult`.
4. **Ambient Drum & Percussion Groove (MIDI Channel 9)**: Synced to the 90 BPM 16th-note clock and toggleable anytime with `[D]` (or disabled at launch via `--no-drums`):
   - Muted heartbeat kick (`36`)
   - Gentle rimshot / side-stick / clap backbeat (`37` / `39`)
   - Crisp closed and open hi-hats (`42` / `46`)
   - Shaker / tambourine / conga accents (`70` / `54` / `63`) that intensify as spatial complexity rises.

---

## 3. How Computer Vision Features Drive the Music

`CVAnalyzer` inspects the active `sensor_size x sensor_size` patch and sends an `AnalysisResult` to `CMinorAudioEngine`. Each metric shapes a distinct musical dimension:

### `dominant_color` -> Register, Timbre & Harmonic Coloring

`CVAnalyzer` classifies the patch into six canonical categories (`Black`, `White`, `Red`, `Orange`, `Green`, `Blue`), capturing deep crimson Fire Weather extremes as `Red`, violet Saharan dust plumes as `Orange`, and applying chromatic salience (`0.18`) so glowing ocean currents or smoke filaments over dark backgrounds remain audible:

| Dominant Color | Register & Scientific Feel | Musical Role |
| :--- | :--- | :--- |
| `Black` | Deepest register (2nd-3rd octave) | Warm dark ambient pad (ocean abyss, night-side Earth, background) |
| `Blue` | Low-mid register (3rd-4th octave) | Atmospheric pad (cold currents, clean air, open ocean) |
| `Green` | Mid register (3rd-4th octave) | Marimba / organic wood timbre (landmasses, sea salt, low fire risk) |
| `White` | Upper-mid register (4th-5th octave) | Vibraphone shimmer (polar ice, cloud decks, baseline snow) |
| `Orange` | Expressive high-mid (4th-5th octave) | Electric piano warmth (elevated CO2, mineral dust, moderate heat) |
| `Red` | Peak upper register (5th-6th octave) | Lead synth highlight (extreme wildfire danger, peak CO2, thermal hotspots) |

### `intensity` (`0.0` to `1.0`) -> Velocity, Brightness & Chord Fullness

- Derived from patch Value and Saturation (`0.65 * mean_v + 0.35 * mean_s`).
- Scales MIDI note velocity (`30` to `127`), MIDI Expression (`CC 11`) / Filter Brightness (`CC 74`), and harmonic chord density.

### `density` (`0.0` to `1.0`) -> Rhythmic Subdivision & Percussion Activity

- Derived from Gaussian-blurred Canny edge pixel ratio normalized against `0.22`.
- Controls the beat-quantized step rate and drum groove fullness:

| Density Range | Rhythmic Subdivision | Typical Visual Feature |
| :--- | :--- | :--- |
| `0.00 - 0.12` | Whole-note pulse (Beat 1) | Calm open ocean, uniform polar ice, flat clean air |
| `0.12 - 0.30` | Half-note pulse (Beats 1 & 3) | Smooth continental basins, broad pressure gradients |
| `0.30 - 0.70` | 8th-note melodic walk | Coastlines, ocean current streams, aerosol fronts |
| `0.70 - 1.00` | 16th-note baroque arpeggio | Turbulent storms, fragmented wildfire clusters, dense grids |

### `shape` (`uniform`, `linear`, `circular`, `organic`) & `shape_score` -> Melodic Contour

Extracted from contour circularity and Sobel gradient / structure-tensor anisotropy:

- `uniform`: Sustained, dreamy organ and pad chords with minimal arpeggio movement.
- `linear`: Directional ascending or descending scalar runs tracing coastlines, jet streams, or sharp fronts.
- `circular`: Cyclical looping baroque arpeggios reflecting compact eddies, hotspots, or enclosed cells.
- `organic`: Expressive syncopated phrasing over irregular natural terrain and patchy plumes.

### `spottiness` (`0.0` to `1.0`) -> Staccato Plucks & Percussion Sparkles

- Measures fragmented multi-spot activity (wildfire clusters, speckled aerosol sources, broken clouds).
- High `spottiness` shortens note gates into crisp staccato plucks, adds syncopated counter-accents, and introduces light conga/tambourine ghost hits on Channel 9.

### `swirliness` (`0.0` to `1.0`) -> Wave Turns, Chorus & Vibrato

- Measures Sobel gradient orientation spread / rotational curl (hurricanes, ECCO2 curling eddies, swirling CO2 plumes).
- High `swirliness` bends the arpeggiator into sweeping pendulum and wave turns while raising MIDI Chorus (`CC 93`) and gentle Modulation (`CC 1`).

### Resizable `sensor_size` (`16` to `160` px, default `40` px) -> Voicing Spread & Spatial Reverb

Adjusted anytime with `[+]` / `[-]` (also `=`, `_`, `[`, `]`) or the mouse scroll wheel:

- **Small cursor (`< 30` px)**: Focuses on fine-scale features (individual hotspots, narrow ocean currents) with tighter, intimate solo voicings and drier articulation.
- **Default cursor (`40` px)**: Balanced regional sampling across coastlines and fronts.
- **Wide cursor (`> 60` px)**: Aggregates broad planetary zones, opening wide-spread multi-octave chord voicings, octave bass/pad doubling, and deeper spatial Reverb (`CC 91`).

---

## 4. Cross-Platform Audio Backends

`create_best_audio_backend(enable_audio=True)` selects the best driver in priority order (`PygameMidiBackend` -> `WinmmMidiBackend` on Windows / `LinuxFluidSynthBackend` on Linux -> `PygameMixerSynthBackend` -> `NullAudioBackend`) without crashing on headless or Linux machines:

1. **`PygameMidiBackend`**: Cross-platform MIDI output via `pygame.midi` when a MIDI port is available.
2. **`WinmmMidiBackend`** (Windows): Native Windows General MIDI synthesizer via `winmm.dll` (loaded lazily on Windows only, zero extra Python dependencies).
3. **`LinuxFluidSynthBackend`** (Linux): Launches FluidSynth software synthesis with SoundFont playback when `fluidsynth` is available on Linux.
4. **`PygameMixerSynthBackend`**: Cross-platform software synthesizer fallback via `pygame.mixer` + NumPy waveforms when hardware/OS MIDI ports are unavailable.
5. **`NullAudioBackend`**: Silent no-op backend used with `--no-audio`, `--self-test`, and headless unit tests.
