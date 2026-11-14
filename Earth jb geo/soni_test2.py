import os
import glob
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import sounddevice as sd

# Window & Audio Constants
WIN_W, WIN_H = 1280, 800
TOP_BAR_H = 58
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
DELAY_SAMPLES = int(SAMPLE_RATE * 0.48)  # 480ms Eno tape-loop delay

# 6 Distinct Chords per Preset (Mapped to Color Spectrum Bands: Red/Orange, Yellow, Green, Cyan, Blue, Purple)
# Preset 0: "Music for Airports" (Lydian / Ambient Major Voicings in Hz)
CHORDS_LIGHT = [
    ("Eb 6/9 (Warm Smoke)", np.array([155.56, 196.00, 233.08, 261.63, 349.23], dtype=np.float64)),
    ("Ab Maj9 (Golden Dust)", np.array([103.83, 155.56, 207.65, 261.63, 311.13], dtype=np.float64)),
    ("Db Maj9 (Green Plume)", np.array([138.59, 174.61, 207.65, 261.63, 311.13], dtype=np.float64)),
    ("Bb min11 (Cyan Swirl)", np.array([116.54, 174.61, 207.65, 261.63, 349.23], dtype=np.float64)),
    ("F min7 (Deep Ocean)", np.array([87.31, 130.81, 174.61, 207.65, 261.63], dtype=np.float64)),
    ("G Maj7#11 (Purple Haze)", np.array([138.59, 196.00, 246.94, 293.66, 392.00], dtype=np.float64)),
]

# Preset 1: "Apollo / Dark Space" (Minor 11th / Suspended Voicings in Hz)
CHORDS_DARK = [
    ("D Phrygian (Red Core)", np.array([73.42, 110.00, 146.83, 155.56, 220.00], dtype=np.float64)),
    ("G Sus4 (Amber Ridge)", np.array([98.00, 130.81, 146.83, 196.00, 261.63], dtype=np.float64)),
    ("C Dorian 11 (Green Canopy)", np.array([65.41, 130.81, 155.56, 196.00, 233.08], dtype=np.float64)),
    ("F Quartal (Cyan Drift)", np.array([87.31, 116.54, 155.56, 207.65, 261.63], dtype=np.float64)),
    ("A min9 (Abyssal Blue)", np.array([55.00, 110.00, 130.81, 164.81, 246.94], dtype=np.float64)),
    ("Eb Maj7#5 (Violet Dust)", np.array([77.78, 155.56, 196.00, 246.94, 293.66], dtype=np.float64)),
]


class SonifierState:
    def __init__(self):
        self.running = True
        self.menu_open = False
        self.is_pressing = False
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.preset = 0
        self.lens_radius = 48

        self.cursor_x = WIN_W // 2
        self.cursor_y = (WIN_H + TOP_BAR_H) // 2

        # 1. COLOUR SPECTRUM -> CHORDS & NOTES
        self.color_spectrum = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.active_chord_idx = 2
        self.chord_name = "Db Maj9 (Green Plume)"
        self.dominant_hue = 0.35
        self.saturation = 0.6

        # 2. PIXEL DENSITY -> RHYTHM
        self.pixel_density = 0.15       # 0.0 (sparse ocean) to 1.0 (dense city lights / storm core)
        self.rhythm_hz = 1.8            # Trigger frequency in Hz
        self.rhythm_phase = 0.0         # Clock accumulator for rhythmic triggers
        self.visual_pulse = 0.0         # UI flash ring synced to rhythm

        # 3. VISUAL SHAPES -> SYNTHESIZER PARAMETERS
        self.shape_elongation = 0.2     # 0 = round blob, 1 = linear streak -> Chorus Detune & Spread
        self.shape_jaggedness = 0.15    # 0 = smooth cloud, 1 = fractal/jagged -> FM Index & Filter Cutoff
        self.shape_angle = 0.5          # 0..1 orientation -> Harmonic Fold & Stereo Pan
        self.shape_label = "Smooth Organic Cloud"

        # Smoothed Synthesis Parameters (Audio Thread)
        self.s_freqs = CHORDS_LIGHT[2][1].copy()
        self.s_density = 0.15
        self.s_elongation = 0.2
        self.s_jaggedness = 0.15
        self.s_angle = 0.5
        self.pad_env = 0.0

        # Oscillator Phase Accumulators
        self.pad_phases = np.zeros(5, dtype=np.float64)
        self.detune_phases = np.zeros(5, dtype=np.float64)
        self.mod_phases = np.zeros(5, dtype=np.float64)

        # Polyphonic Rhythmic Pluck / Bell Voices (4 voices)
        self.pluck_freqs = np.array([261.63, 311.13, 349.23, 523.25], dtype=np.float64)
        self.pluck_envs = np.zeros(4, dtype=np.float64)
        self.pluck_phases = np.zeros(4, dtype=np.float64)
        self.pluck_pans = np.array([0.2, 0.8, 0.35, 0.65], dtype=np.float64)
        self.next_pluck = 0
        self.arp_step = 0

        # Low-pass filter state & Stereo Delay Buffer
        self.lp_L = 0.0
        self.lp_R = 0.0
        self.delay_L = np.zeros(DELAY_SAMPLES, dtype=np.float32)
        self.delay_R = np.zeros(DELAY_SAMPLES, dtype=np.float32)
        self.delay_ptr = 0


state = SonifierState()


def generate_aerosol_fallback():
    """Generates a synthetic aerosol & city-light map if the local file is missing."""
    h, w = 1080, 1920
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), (18, 22, 18), dtype=np.uint8)

    # Green plumes
    green_wave = np.sin(x / 160.0 + np.cos(y / 130.0) * 2.0) * np.cos(y / 200.0)
    g_mask = np.clip((green_wave - 0.15) * 2.2, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] + g_mask * 220, 0, 255).astype(np.uint8)

    # Purple dust band
    purple_band = np.exp(-((y - h * 0.42 + np.sin(x / 190.0) * 65.0) ** 2) / (2 * 80.0**2))
    p_mask = purple_band * np.clip(np.sin(x / 380.0), 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] + p_mask * 210, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + p_mask * 115, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] + p_mask * 205, 0, 255).astype(np.uint8)

    # Orange/red smoke core
    smoke = np.exp(-((x - w * 0.54) ** 2 + (y - h * 0.58) ** 2) / (2 * 95.0**2))
    img[:, :, 2] = np.clip(img[:, :, 2] + smoke * 245, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + smoke * 95, 0, 255).astype(np.uint8)

    # High-density city light clusters for rhythmic testing
    rng = np.random.default_rng(42)
    for cx, cy in [(450, 340), (980, 290), (1420, 420)]:
        pts_x = rng.normal(cx, 55, 900).astype(int)
        pts_y = rng.normal(cy, 35, 900).astype(int)
        valid = (pts_x >= 0) & (pts_x < w) & (pts_y >= 0) & (pts_y < h)
        img[pts_y[valid], pts_x[valid]] = [210, 250, 255]

    return img


class ImageGallery:
    def __init__(self):
        self.entries = []
        self.current_idx = 0
        self.centered_canvas = None
        self.img_rect = (0, TOP_BAR_H, WIN_W, WIN_H - TOP_BAR_H)

        self.zoom = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 12.0
        self.center_u = 0.5
        self.center_v = 0.5

        self.load_initial_images()

    def load_initial_images(self):
        target_name = "Aerosols_2025-08-19T20_00_00_5760x3240.jpg"
        loaded_paths = set()

        if os.path.exists(target_name):
            img = cv2.imread(target_name)
            if img is not None:
                self.entries.append(("NASA Global Aerosols", img))
                loaded_paths.add(os.path.abspath(target_name))

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for path in sorted(glob.glob(ext)):
                abs_p = os.path.abspath(path)
                if abs_p not in loaded_paths:
                    img = cv2.imread(path)
                    if img is not None:
                        label = os.path.splitext(os.path.basename(path))[0][:28]
                        self.entries.append((label, img))
                        loaded_paths.add(abs_p)

        if not self.entries:
            self.entries.append(("NASA Global Aerosols (Built-in)", generate_aerosol_fallback()))
        self.entries.append(("Geometric Synth Test Map", self._make_shape_test_map()))

        self.select_image(0)

    def _make_shape_test_map(self):
        """Creates a map with distinct colors, pixel densities, and geometric shapes."""
        h, w = 1080, 1920
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        b = np.clip(125 + 115 * np.sin(x / 170.0 + y / 220.0), 0, 255)
        g = np.clip(110 + 115 * np.cos(x / 120.0 - y / 150.0), 0, 255)
        r = np.clip(115 + 125 * np.sin((x - y) / 190.0), 0, 255)
        img = np.dstack([b, g, r]).astype(np.uint8)
        # Add linear stripes and concentric circles to test shape parameters
        stripes = ((x + y) % 24 < 4).astype(np.uint8) * 110
        img[:, :, 1] = np.clip(img[:, :, 1] + stripes, 0, 255)
        return img

    def select_image(self, idx):
        self.current_idx = idx % len(self.entries)
        self.reset_view()

    def reset_view(self):
        self.zoom = 1.0
        self.center_u = 0.5
        self.center_v = 0.5
        self.render_viewport()

    def zoom_at(self, factor, screen_x=None, screen_y=None):
        old_zoom = self.zoom
        new_zoom = float(np.clip(old_zoom * factor, self.min_zoom, self.max_zoom))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        ix, iy, iw, ih = self.img_rect
        if screen_x is not None and screen_y is not None and iw > 0 and ih > 0:
            rel_x = np.clip((screen_x - ix) / iw, 0.0, 1.0) - 0.5
            rel_y = np.clip((screen_y - iy) / ih, 0.0, 1.0) - 0.5
            self.center_u += rel_x * (1.0 / old_zoom - 1.0 / new_zoom)
            self.center_v += rel_y * (1.0 / old_zoom - 1.0 / new_zoom)

        self.zoom = new_zoom
        self._clamp_center()
        self.render_viewport()

    def pan_by_pixels(self, dx_screen, dy_screen):
        _, _, iw, ih = self.img_rect
        if iw > 0 and ih > 0 and self.zoom > 1.0:
            self.center_u -= (dx_screen / iw) / self.zoom
            self.center_v -= (dy_screen / ih) / self.zoom
            self._clamp_center()
            self.render_viewport()

    def _clamp_center(self):
        half_span = 0.5 / self.zoom
        self.center_u = float(np.clip(self.center_u, half_span, 1.0 - half_span))
        self.center_v = float(np.clip(self.center_v, half_span, 1.0 - half_span))

    def render_viewport(self):
        _, raw_img = self.entries[self.current_idx]
        ih, iw = raw_img.shape[:2]

        avail_w = WIN_W - 40
        avail_h = WIN_H - TOP_BAR_H - 48
        scale = min(avail_w / iw, avail_h / ih)
        disp_w, disp_h = max(1, int(iw * scale)), max(1, int(ih * scale))

        crop_w = max(8, int(iw / self.zoom))
        crop_h = max(8, int(ih / self.zoom))
        cx = int(self.center_u * iw)
        cy = int(self.center_v * ih)

        x1 = int(np.clip(cx - crop_w // 2, 0, iw - crop_w))
        y1 = int(np.clip(cy - crop_h // 2, 0, ih - crop_h))
        cropped = raw_img[y1 : y1 + crop_h, x1 : x1 + crop_w]

        interp = cv2.INTER_AREA if self.zoom < 2.5 else cv2.INTER_LINEAR
        resized = cv2.resize(cropped, (disp_w, disp_h), interpolation=interp)

        canvas = np.full((WIN_H, WIN_W, 3), (14, 15, 18), dtype=np.uint8)
        x_off = (WIN_W - disp_w) // 2
        y_off = TOP_BAR_H + (WIN_H - TOP_BAR_H - disp_h) // 2
        canvas[y_off : y_off + disp_h, x_off : x_off + disp_w] = resized

        cv2.rectangle(
            canvas,
            (x_off - 1, y_off - 1),
            (x_off + disp_w + 1, y_off + disp_h + 1),
            (65, 70, 80),
            1,
        )
        self.centered_canvas = canvas
        self.img_rect = (x_off, y_off, disp_w, disp_h)

    def add_image_via_dialog(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Satellite / Map Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")],
        )
        root.destroy()
        if file_path and os.path.exists(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                label = os.path.splitext(os.path.basename(file_path))[0][:28]
                self.entries.append((label, img))
                self.select_image(len(self.entries) - 1)
                state.menu_open = False


def analyze_lens(canvas, x, y, radius, img_rect):
    """
    Extracts the 3 core mappings inside the circular lens:
    1. Colour Spectrum -> Chords & Notes
    2. Pixel Density -> Rhythm (BPM / Trigger Rate)
    3. Visual Shapes -> Synthesizer Timbre Parameters
    """
    ix, iy, iw, ih = img_rect
    cx = int(np.clip(x, ix, ix + iw - 1))
    cy = int(np.clip(y, iy, iy + ih - 1))

    x1, x2 = max(ix, cx - radius), min(ix + iw, cx + radius)
    y1, y2 = max(iy, cy - radius), min(iy + ih, cy + radius)
    roi = canvas[y1:y2, x1:x2]
    if roi.size == 0:
        return

    rh, rw = roi.shape[:2]
    Y, X = np.ogrid[:rh, :rw]
    mask = (X - (cx - x1)) ** 2 + (Y - (cy - y1)) ** 2 <= radius**2
    if not np.any(mask):
        return

    # =========================================================================
    # 1. COLOUR SPECTRUMS -> CHORDS & NOTES
    # =========================================================================
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_vals = hsv_roi[:, :, 0][mask].astype(np.float32)  # 0..179 in OpenCV
    s_vals = hsv_roi[:, :, 1][mask].astype(np.float32) / 255.0
    v_vals = hsv_roi[:, :, 2][mask].astype(np.float32) / 255.0

    # Weight each pixel by its chromatic prominence so dark ocean doesn't flatten chords
    chroma_weight = s_vals * np.clip(v_vals * 1.5, 0.05, 1.0)

    # 6 Hue Spectrum Bins:
    # 0: Red/Orange (0-20 & 165-180), 1: Yellow (20-40), 2: Green (40-85),
    # 3: Cyan (85-108), 4: Blue (108-135), 5: Purple/Magenta (135-165)
    bins = np.zeros(6, dtype=np.float64)
    bins[0] = np.sum(chroma_weight[(h_vals < 20) | (h_vals >= 165)])
    bins[1] = np.sum(chroma_weight[(h_vals >= 20) & (h_vals < 40)])
    bins[2] = np.sum(chroma_weight[(h_vals >= 40) & (h_vals < 85)])
    bins[3] = np.sum(chroma_weight[(h_vals >= 85) & (h_vals < 108)])
    bins[4] = np.sum(chroma_weight[(h_vals >= 108) & (h_vals < 135)])
    bins[5] = np.sum(chroma_weight[(h_vals >= 135) & (h_vals < 165)])

    total_chroma = np.sum(bins)
    if total_chroma > 0.5:
        spectrum = bins / total_chroma
        dom_idx = int(np.argmax(spectrum))
    else:
        # Very dark/desaturated region defaults to deep Blue/Ocean chord
        spectrum = np.array([0.05, 0.05, 0.1, 0.1, 0.6, 0.1], dtype=np.float64)
        dom_idx = 4

    state.color_spectrum = spectrum
    state.active_chord_idx = dom_idx
    chord_bank = CHORDS_LIGHT if state.preset == 0 else CHORDS_DARK
    state.chord_name = chord_bank[dom_idx][0]

    # Blend the chord frequencies weighted by the top 2 spectrum bands for rich harmonic transitions
    top2 = np.argsort(spectrum)[-2:]
    w1, w2 = spectrum[top2[1]], spectrum[top2[0]]
    if w1 + w2 > 0:
        target_freqs = (chord_bank[top2[1]][1] * w1 + chord_bank[top2[0]][1] * w2) / (w1 + w2)
    else:
        target_freqs = chord_bank[dom_idx][1]
    state.s_freqs = 0.75 * state.s_freqs + 0.25 * target_freqs

    # Dominant hue for UI ring coloring
    hue_centers = [0.03, 0.16, 0.35, 0.53, 0.67, 0.83]
    state.dominant_hue = hue_centers[dom_idx]
    state.saturation = float(np.clip(np.mean(s_vals) * 1.6, 0.2, 1.0))

    # =========================================================================
    # 2. PIXEL DENSITY -> RHYTHM (BPM / Pulse Rate)
    # =========================================================================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Combine luminous pixel ratio + local high-frequency stipple (city lights / dense aerosols)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    stipple = cv2.absdiff(gray, blur)[mask]
    stipple_ratio = float(np.mean(stipple > 10))
    bright_ratio = float(np.mean(v_vals > 0.32))
    variance_norm = float(np.clip(np.std(v_vals) * 3.2, 0.0, 1.0))

    density = float(np.clip(bright_ratio * 0.45 + stipple_ratio * 1.8 + variance_norm * 0.35, 0.0, 1.0))
    state.pixel_density = density
    # Map density 0..1 to Rhythm Rate: 0.9 Hz (54 BPM slow drift) up to 9.5 Hz (570 BPM fast sequencer)
    state.rhythm_hz = 0.9 + (density ** 1.35) * 8.6

    # =========================================================================
    # 3. VISUAL SHAPES -> SYNTHESIZER PARAMETERS
    # =========================================================================
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Structure Tensor eigenvalues for Shape Elongation (Streaks/Filaments vs Round Blobs)
    jxx = float(np.mean(gx[mask] ** 2)) + 1e-5
    jyy = float(np.mean(gy[mask] ** 2)) + 1e-5
    jxy = float(np.mean(gx[mask] * gy[mask]))

    trace = jxx + jyy
    det_term = np.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * (jxy ** 2)))
    lambda1 = 0.5 * (trace + det_term)
    lambda2 = 0.5 * (trace - det_term)

    # Coherence: 0 = isotropic circular shape, 1 = strongly aligned linear streak
    elongation = float(np.clip(((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-5)) * 1.4, 0.0, 1.0))

    # Dominant shape angle (0..1)
    angle_rad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    shape_angle = float((angle_rad + np.pi / 2.0) / np.pi)

    # Shape Jaggedness / Contour Complexity via Laplacian & edge perimeter density
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))[mask]
    jaggedness = float(np.clip(np.mean(lap) / 65.0, 0.0, 1.0))

    state.shape_elongation = elongation
    state.shape_jaggedness = jaggedness
    state.shape_angle = shape_angle

    # Descriptive shape classification for the HUD
    if jaggedness > 0.45:
        state.shape_label = "Jagged / Fractal Ridge (High FM)"
    elif elongation > 0.52:
        state.shape_label = "Linear Filament / Jet (Wide Chorus)"
    elif density > 0.55:
        state.shape_label = "Circular Vortex / Cluster (Warm Resonant)"
    else:
        state.shape_label = "Smooth Organic Cloud (Pure Sine Pad)"


def audio_callback(outdata, frames, time_info, status):
    """
    Real-Time Synthesis Engine:
    - Chords & Notes driven by Colour Spectrum
    - Rhythmic Pulse & Arpeggiator driven by Pixel Density
    - FM Index, Chorus Detune, and Filter Cutoff driven by Visual Shapes
    """
    a = 0.12
    state.s_density += a * (state.pixel_density - state.s_density)
    state.s_elongation += a * (state.shape_elongation - state.s_elongation)
    state.s_jaggedness += a * (state.shape_jaggedness - state.s_jaggedness)
    state.s_angle += a * (state.shape_angle - state.s_angle)

    # Hold-to-play envelope
    target = 1.0 if (state.is_pressing and not state.menu_open) else 0.0
    rate = 0.0020 if target > 0.5 else 0.00038
    t_idx = np.arange(1, frames + 1, dtype=np.float32)
    env = state.pad_env + (target - state.pad_env) * (1.0 - (1.0 - rate) ** t_idx)
    state.pad_env = float(env[-1])

    # =========================================================================
    # RHYTHM GENERATOR (Driven by Pixel Density)
    # =========================================================================
    rhythm_inc = state.rhythm_hz / SAMPLE_RATE
    r_phases = state.rhythm_phase + np.arange(1, frames + 1, dtype=np.float64) * rhythm_inc
    # Count integer boundaries crossed in this audio block to trigger rhythmic notes
    triggers = np.floor(r_phases) > np.floor(np.insert(r_phases[:-1], 0, state.rhythm_phase))
    state.rhythm_phase = float(r_phases[-1] % 1.0)
    state.visual_pulse = float(np.exp(-state.rhythm_phase * 4.0)) if state.is_pressing else 0.0

    if np.any(triggers) and state.is_pressing and not state.menu_open:
        slot = state.next_pluck
        chord_bank = CHORDS_LIGHT if state.preset == 0 else CHORDS_DARK
        active_notes = chord_bank[state.active_chord_idx][1]

        # Step through chord degrees weighted by color spectrum
        note_idx = state.arp_step % len(active_notes)
        state.arp_step += 1
        octave = 2.0 if (state.s_density > 0.45 and state.arp_step % 2 == 0) else 1.0

        state.pluck_freqs[slot] = active_notes[note_idx] * octave
        state.pluck_envs[slot] = min(1.0, 0.35 + state.s_density * 0.55)
        state.pluck_pans[slot] = float(np.clip(0.5 + 0.4 * np.sin(state.arp_step * 1.7), 0.1, 0.9))
        state.next_pluck = (slot + 1) % 4

    # Rhythmic pulse modulation on the pad (subtle at low density, pronounced pulse at high density)
    pulse_lfo = 1.0 - (0.15 + 0.45 * state.s_density) * (0.5 - 0.5 * np.cos(2.0 * np.pi * r_phases))

    # =========================================================================
    # SYNTHESIZER TIMBRE (Driven by Visual Shapes)
    # =========================================================================
    # Shape Elongation -> Detune spread (Hz)
    detune_spread = 0.001 + state.s_elongation * 0.012
    # Shape Jaggedness -> FM Metallic Index
    fm_index = state.s_jaggedness * 3.4
    # Shape Orientation Angle -> Modulator harmonic ratio (1x warm vs 2x bright octave)
    mod_ratio = 1.0 if state.s_angle < 0.55 else 2.0

    pad_L = np.zeros(frames, dtype=np.float64)
    pad_R = np.zeros(frames, dtype=np.float64)

    voice_gains = [0.32, 0.25, 0.22, 0.18, 0.16]
    for i in range(5):
        f_base = state.s_freqs[i]
        inc_main = (2.0 * np.pi * f_base) / SAMPLE_RATE
        inc_det = (2.0 * np.pi * f_base * (1.0 + detune_spread)) / SAMPLE_RATE
        inc_mod = (2.0 * np.pi * f_base * mod_ratio) / SAMPLE_RATE

        p_main = state.pad_phases[i] + t_idx * inc_main
        p_det = state.detune_phases[i] + t_idx * inc_det
        p_mod = state.mod_phases[i] + t_idx * inc_mod

        state.pad_phases[i] = float(p_main[-1] % (2.0 * np.pi))
        state.detune_phases[i] = float(p_det[-1] % (2.0 * np.pi))
        state.mod_phases[i] = float(p_mod[-1] % (2.0 * np.pi))

        mod_sig = np.sin(p_mod) * fm_index
        osc_main = np.sin(p_main + mod_sig)
        osc_det = np.sin(p_det - mod_sig * 0.5)

        pan = 0.2 + 0.15 * i
        gain = voice_gains[i]
        pad_L += (osc_main * (1.0 - pan) + osc_det * pan) * gain
        pad_R += (osc_main * pan + osc_det * (1.0 - pan)) * gain

    pad_L = pad_L * env * pulse_lfo
    pad_R = pad_R * env * pulse_lfo

    # Render Rhythmic Pluck Voices (Decay speed scales with Pixel Density so fast rhythms stay crisp)
    pluck_L = np.zeros(frames, dtype=np.float64)
    pluck_R = np.zeros(frames, dtype=np.float64)
    decay_rate = 0.9994 - state.s_density * 0.00045
    decay_curve = decay_rate ** t_idx

    for i in range(4):
        if state.pluck_envs[i] > 0.001:
            p_env = state.pluck_envs[i] * decay_curve
            state.pluck_envs[i] = float(p_env[-1])
            p_inc = (2.0 * np.pi * state.pluck_freqs[i]) / SAMPLE_RATE
            p_phase = state.pluck_phases[i] + t_idx * p_inc
            state.pluck_phases[i] = float(p_phase[-1] % (2.0 * np.pi))

            # Shape jaggedness makes plucks brighter/bell-like; smooth shapes make soft marimba/sine plucks
            bell_fm = (0.4 + state.s_jaggedness * 2.8) * p_env
            tone = np.sin(p_phase + bell_fm * np.sin(p_phase * 3.0)) * p_env * 0.24
            pan = state.pluck_pans[i]
            pluck_L += tone * (1.0 - pan)
            pluck_R += tone * pan

    raw_L = (pad_L * 0.28 + pluck_L).astype(np.float32)
    raw_R = (pad_R * 0.28 + pluck_R).astype(np.float32)

    # One-pole Low-Pass Filter controlled by Visual Shape + Density
    cutoff = float(np.clip(0.08 + state.s_jaggedness * 0.45 + state.s_elongation * 0.25, 0.06, 0.85))
    filt_L = np.zeros(frames, dtype=np.float32)
    filt_R = np.zeros(frames, dtype=np.float32)
    lp_l, lp_r = state.lp_L, state.lp_R
    for n in range(frames):
        lp_l += cutoff * (raw_L[n] - lp_l)
        lp_r += cutoff * (raw_R[n] - lp_r)
        filt_L[n] = lp_l
        filt_R[n] = lp_r
    state.lp_L, state.lp_R = float(lp_l), float(lp_r)

    # Stereo Ping-Pong Tape Delay
    idx = (state.delay_ptr + np.arange(frames)) % DELAY_SAMPLES
    del_L = state.delay_L[idx]
    del_R = state.delay_R[idx]

    fb = 0.62
    state.delay_L[idx] = np.tanh(filt_L + del_R * fb)
    state.delay_R[idx] = np.tanh(filt_R + del_L * fb)
    state.delay_ptr = int((state.delay_ptr + frames) % DELAY_SAMPLES)

    outdata[:, 0] = np.tanh(filt_L + del_L * 0.52)
    outdata[:, 1] = np.tanh(filt_R + del_R * 0.52)


def draw_interface(gallery):
    """Renders the centered image, top bar, live 3-part mapping HUD, and reactive lens."""
    frame = gallery.centered_canvas.copy()

    # Top Navigation Bar
    cv2.rectangle(frame, (0, 0), (WIN_W, TOP_BAR_H), (22, 24, 30), -1)
    cv2.line(frame, (0, TOP_BAR_H), (WIN_W, TOP_BAR_H), (60, 65, 75), 1)

    menu_col = (95, 150, 245) if state.menu_open else (45, 52, 65)
    cv2.rectangle(frame, (16, 10), (195, 48), menu_col, -1, cv2.LINE_AA)
    cv2.putText(frame, "Images Menu [M]", (28, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (207, 10), (370, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, "+ Add Picture [O]", (218, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 235, 210), 1, cv2.LINE_AA)

    p_name = "Palette: Eno Airports [SPACE]" if state.preset == 0 else "Palette: Dark Apollo [SPACE]"
    p_col = (70, 95, 65) if state.preset == 0 else (85, 55, 95)
    cv2.rectangle(frame, (382, 10), (650, 48), p_col, -1, cv2.LINE_AA)
    cv2.putText(frame, p_name, (394, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (662, 10), (800, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, f"Zoom: {gallery.zoom:.1f}x [R]", (672, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 220, 150), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (WIN_W - 120, 10), (WIN_W - 16, 48), (60, 45, 155), -1, cv2.LINE_AA)
    cv2.putText(frame, "Quit [Q]", (WIN_W - 98, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if not state.menu_open:
        # Draw Live 3-Mapping Telemetry Panel (Top-Left of Viewport)
        panel_x, panel_y = 22, TOP_BAR_H + 14
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 415, panel_y + 108), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 415, panel_y + 108), (70, 75, 90), 1)

        # 1. Colour -> Chord readout
        cv2.putText(
            frame,
            f"1. COLOR -> CHORD: {state.chord_name}",
            (panel_x + 12, panel_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (150, 240, 190),
            1,
            cv2.LINE_AA,
        )
        # 2. Pixel Density -> Rhythm readout
        bpm = int(state.rhythm_hz * 60)
        cv2.putText(
            frame,
            f"2. DENSITY -> RHYTHM: {state.rhythm_hz:.1f} Hz ({bpm} BPM) | Density: {int(state.pixel_density*100)}%",
            (panel_x + 12, panel_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (250, 220, 140),
            1,
            cv2.LINE_AA,
        )
        # 3. Visual Shapes -> Synth Parameters readout
        cv2.putText(
            frame,
            f"3. SHAPE -> SYNTH: {state.shape_label}",
            (panel_x + 12, panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (170, 210, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw Reactive Cursor Lens
        hsv_swatch = np.uint8([[[int(state.dominant_hue * 179), int(state.saturation * 255), 245]]])
        ring_bgr = tuple(int(c) for c in cv2.cvtColor(hsv_swatch, cv2.COLOR_HSV2BGR)[0, 0])

        # Outer ring pulses in sync with the Pixel Density Rhythm Clock
        rhythm_expand = int(state.visual_pulse * 12)
        cv2.circle(
            frame,
            (state.cursor_x, state.cursor_y),
            state.lens_radius + rhythm_expand,
            ring_bgr,
            2,
            cv2.LINE_AA,
        )

        # Inner ellipse morphs to show detected Visual Shape (Elongation & Angle)
        ax1 = int(max(6, state.lens_radius * 0.55))
        ax2 = int(max(4, ax1 * (1.0 - state.shape_elongation * 0.65)))
        ang_deg = int(state.shape_angle * 180.0)
        cv2.ellipse(
            frame,
            (state.cursor_x, state.cursor_y),
            (ax1, ax2),
            ang_deg,
            0,
            360,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        title = gallery.entries[gallery.current_idx][0]
        cv2.putText(
            frame,
            f"Map: {title} | Scroll or [+/-] Zoom | Right-Drag or [Arrows] Pan | [R] Reset | [W/S] Lens | [Q] Quit",
            (20, WIN_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (165, 170, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        # Image Selection Menu Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (140, 95), (WIN_W - 140, WIN_H - 75), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (140, 95), (WIN_W - 140, WIN_H - 75), (90, 120, 180), 2)

        cv2.putText(
            frame,
            "Select an Image to Sonify (Click card or press 1-9, or press [O] to browse files)",
            (175, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )

        cols, card_w, card_h = 3, 280, 210
        start_x, start_y, gap_x, gap_y = 180, 165, 40, 35

        for idx, (label, img) in enumerate(gallery.entries[:6]):
            r, c = divmod(idx, cols)
            cx = start_x + c * (card_w + gap_x)
            cy = start_y + r * (card_h + gap_y)

            thumb = cv2.resize(img, (card_w, card_h - 38), interpolation=cv2.INTER_AREA)
            frame[cy : cy + card_h - 38, cx : cx + card_w] = thumb

            border_col = (100, 230, 140) if idx == gallery.current_idx else (75, 80, 95)
            cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), border_col, 2)
            cv2.rectangle(frame, (cx, cy + card_h - 38), (cx + card_w, cy + card_h), (32, 36, 46), -1)
            cv2.putText(
                frame,
                f"[{idx + 1}] {label[:24]}",
                (cx + 10, cy + card_h - 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    return frame


def make_mouse_handler(gallery):
    def on_mouse(event, x, y, flags, param):
        state.cursor_x, state.cursor_y = x, y

        if event == cv2.EVENT_MOUSEWHEEL and not state.menu_open:
            if flags > 0:
                gallery.zoom_at(1.22, x, y)
            else:
                gallery.zoom_at(1.0 / 1.22, x, y)
            if state.is_pressing:
                analyze_lens(gallery.centered_canvas, x, y, state.lens_radius, gallery.img_rect)
            return

        if event == cv2.EVENT_RBUTTONDOWN and not state.menu_open:
            state.is_panning = True
            state.pan_start_x, state.pan_start_y = x, y
            return
        elif event == cv2.EVENT_RBUTTONUP:
            state.is_panning = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if y <= TOP_BAR_H:
                if 16 <= x <= 195:
                    state.menu_open = not state.menu_open
                elif 207 <= x <= 370:
                    gallery.add_image_via_dialog()
                elif 382 <= x <= 650:
                    state.preset = 1 - state.preset
                elif 662 <= x <= 800:
                    gallery.reset_view()
                elif x >= WIN_W - 120:
                    state.running = False
                return

            if state.menu_open:
                cols, card_w, card_h = 3, 280, 210
                start_x, start_y, gap_x, gap_y = 180, 165, 40, 35
                for idx in range(min(6, len(gallery.entries))):
                    r, c = divmod(idx, cols)
                    cx = start_x + c * (card_w + gap_x)
                    cy = start_y + r * (card_h + gap_y)
                    if cx <= x <= cx + card_w and cy <= y <= cy + card_h:
                        gallery.select_image(idx)
                        state.menu_open = False
                        return
                return

            state.is_pressing = True
            analyze_lens(gallery.centered_canvas, x, y, state.lens_radius, gallery.img_rect)

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.is_panning and not state.menu_open:
                dx = x - state.pan_start_x
                dy = y - state.pan_start_y
                state.pan_start_x, state.pan_start_y = x, y
                gallery.pan_by_pixels(dx, dy)
            elif state.is_pressing and not state.menu_open:
                analyze_lens(gallery.centered_canvas, x, y, state.lens_radius, gallery.img_rect)

        elif event == cv2.EVENT_LBUTTONUP:
            state.is_pressing = False

    return on_mouse


def main():
    gallery = ImageGallery()
    win_name = "Brian Eno Satellite Sonifier"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, make_mouse_handler(gallery))

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype="float32",
        callback=audio_callback,
    )

    with stream:
        while state.running:
            frame = draw_interface(gallery)
            cv2.imshow(win_name, frame)

            key = cv2.waitKeyEx(16)
            if key == -1:
                continue

            low_key = key & 0xFF
            if low_key in (ord("q"), ord("Q"), 27):
                break
            elif low_key in (ord("m"), ord("M")):
                state.menu_open = not state.menu_open
            elif low_key in (ord("o"), ord("O")):
                gallery.add_image_via_dialog()
            elif low_key == ord(" "):
                state.preset = 1 - state.preset
            elif low_key in (ord("r"), ord("R")):
                gallery.reset_view()
            elif low_key in (ord("="), ord("+"), ord("z"), ord("Z")):
                gallery.zoom_at(1.25, state.cursor_x, state.cursor_y)
            elif low_key in (ord("-"), ord("_"), ord("x"), ord("X")):
                gallery.zoom_at(1.0 / 1.25, state.cursor_x, state.cursor_y)
            elif low_key in (ord("w"), ord("W")):
                state.lens_radius = min(130, state.lens_radius + 6)
            elif low_key in (ord("s"), ord("S")):
                state.lens_radius = max(14, state.lens_radius - 6)
            elif ord("1") <= low_key <= ord("9"):
                idx = low_key - ord("1")
                if idx < len(gallery.entries):
                    gallery.select_image(idx)
                    state.menu_open = False
            elif key in (2424832, 65361) or low_key == ord("a"):
                gallery.pan_by_pixels(60, 0)
            elif key in (2555904, 65363) or low_key == ord("d"):
                gallery.pan_by_pixels(-60, 0)
            elif key in (2490368, 65362):
                gallery.pan_by_pixels(0, 60)
            elif key in (2621440, 65364):
                gallery.pan_by_pixels(0, -60)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()