import os
import glob
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import sounddevice as sd

# Detect Monitor Resolution for True Full Screen
_tk_root = tk.Tk()
_tk_root.withdraw()
WIN_W = max(1280, _tk_root.winfo_screenwidth())
WIN_H = max(800, _tk_root.winfo_screenheight())
_tk_root.destroy()

TOP_BAR_H = 58
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
DELAY_SAMPLES = int(SAMPLE_RATE * 0.25)  # Subtle, tight 250ms room reflection

# Harmonious Melodic Chords (All in the same Key so every note harmonizes)
# Each entry: (Label, [Root, 3rd, 5th, Octave, 7th_Alert_Note] in Hz)
# For cool colors, only the pure triad + octave play; Red/Warm zones blend in the 5th element (7th interval).
CHORDS_WARM_KEY = [
    # 0: Red/Orange Zone of Interest -> G7 / Dmaj7 Warm Alert 7th Chord
    ("G7 Alert (Red Hotspot)", np.array([196.00, 246.94, 293.66, 392.00, 349.23], dtype=np.float64)),
    # 1: Yellow/Amber -> D minor7 / F6 Warmth
    ("Dm7 (Amber Ridge)", np.array([146.83, 174.61, 220.00, 293.66, 261.63], dtype=np.float64)),
    # 2: Green Plumes -> F Major Pure Triad
    ("F Major (Green Plume)", np.array([174.61, 220.00, 261.63, 349.23, 329.63], dtype=np.float64)),
    # 3: Cyan Swirls -> A minor Pure Triad
    ("A Minor (Cyan Swirl)", np.array([110.00, 164.81, 220.00, 261.63, 329.63], dtype=np.float64)),
    # 4: Blue Ocean -> C Major Home Triad
    ("C Major (Deep Ocean)", np.array([130.81, 164.81, 196.00, 261.63, 246.94], dtype=np.float64)),
    # 5: Purple Dust -> E minor / G Major Pure Triad
    ("G Major (Purple Dust)", np.array([146.83, 196.00, 246.94, 293.66, 329.63], dtype=np.float64)),
]

CHORDS_DEEP_KEY = [
    # 0: Red/Orange Zone of Interest -> C7 / A7 Warm Alert 7th Chord
    ("C7 Alert (Red Hotspot)", np.array([130.81, 164.81, 196.00, 261.63, 233.08], dtype=np.float64)),
    # 1: Yellow/Amber -> Gm7 Warmth
    ("Gm7 (Amber Ridge)", np.array([98.00, 116.54, 146.83, 196.00, 174.61], dtype=np.float64)),
    # 2: Green Plumes -> Bb Major Pure Triad
    ("Bb Major (Green Plume)", np.array([116.54, 146.83, 174.61, 233.08, 220.00], dtype=np.float64)),
    # 3: Cyan Swirls -> D minor Pure Triad
    ("D Minor (Cyan Swirl)", np.array([73.42, 110.00, 146.83, 174.61, 220.00], dtype=np.float64)),
    # 4: Blue Ocean -> F Major Home Triad
    ("F Major (Deep Ocean)", np.array([87.31, 110.00, 130.81, 174.61, 164.81], dtype=np.float64)),
    # 5: Purple Dust -> C Major Pure Triad
    ("C Major (Purple Dust)", np.array([98.00, 130.81, 164.81, 196.00, 220.00], dtype=np.float64)),
]


class SonifierState:
    def __init__(self):
        self.running = True
        self.fullscreen = True
        self.menu_open = False
        self.is_pressing = False
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.preset = 0
        self.lens_radius = 48

        self.cursor_x = WIN_W // 2
        self.cursor_y = (WIN_H + TOP_BAR_H) // 2

        # 1. COLOUR SPECTRUMS -> HARMONIOUS CHORDS & WARM 7TH ALERT
        self.active_chord_idx = 4
        self.chord_name = "C Major (Deep Ocean)"
        self.warm_alert_7th = 0.0   # 0.0 in cool zones, rises to 1.0 in bright Red/Orange zones
        self.dominant_hue = 0.60
        self.saturation = 0.6

        # 2. PIXEL DENSITY -> MELODIC RHYTHM
        self.pixel_density = 0.15
        self.rhythm_hz = 2.0
        self.rhythm_phase = 0.0
        self.visual_pulse = 0.0

        # 3. VISUAL SHAPES -> CLEAN SYNTH PARAMETERS (No Wobble)
        self.shape_elongation = 0.2   # Controls note envelope crispness (Legato vs Pluck)
        self.shape_jaggedness = 0.15  # Controls warm filter cutoff & octave harmonic brightness
        self.shape_angle = 0.5        # Controls melody arpeggio direction & stereo balance
        self.shape_label = "Smooth Pad (Warm Filter)"

        # Smoothed Audio Thread Parameters
        self.s_density = 0.15
        self.s_alert_7th = 0.0
        self.s_elongation = 0.2
        self.s_jaggedness = 0.15
        self.s_angle = 0.5
        self.pad_env = 0.0

        # Pitch-Locked Oscillator Phases (5 chord voices: Root, 3rd, 5th, Oct, 7th_Alert)
        self.pad_phases = np.zeros(5, dtype=np.float64)
        self.oct_phases = np.zeros(5, dtype=np.float64)

        # Clean Melodic Note Voices (6 polyphonic voices for smooth overlapping melody notes)
        self.note_freqs = np.full(6, 261.63, dtype=np.float64)
        self.note_envs = np.zeros(6, dtype=np.float64)
        self.note_phases = np.zeros(6, dtype=np.float64)
        self.note_pans = np.array([0.3, 0.7, 0.4, 0.6, 0.35, 0.65], dtype=np.float64)
        self.next_note_slot = 0
        self.melody_step = 0

        # Warm Low-Pass Filter & Subtle Room Reflection Buffer
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

    green_wave = np.sin(x / 160.0 + np.cos(y / 130.0) * 2.0) * np.cos(y / 200.0)
    g_mask = np.clip((green_wave - 0.15) * 2.2, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] + g_mask * 220, 0, 255).astype(np.uint8)

    purple_band = np.exp(-((y - h * 0.42 + np.sin(x / 190.0) * 65.0) ** 2) / (2 * 80.0**2))
    p_mask = purple_band * np.clip(np.sin(x / 380.0), 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] + p_mask * 210, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + p_mask * 115, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] + p_mask * 205, 0, 255).astype(np.uint8)

    smoke = np.exp(-((x - w * 0.54) ** 2 + (y - h * 0.58) ** 2) / (2 * 95.0**2))
    img[:, :, 2] = np.clip(img[:, :, 2] + smoke * 245, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + smoke * 95, 0, 255).astype(np.uint8)

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
        h, w = 1080, 1920
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        b = np.clip(125 + 115 * np.sin(x / 170.0 + y / 220.0), 0, 255)
        g = np.clip(110 + 115 * np.cos(x / 120.0 - y / 150.0), 0, 255)
        r = np.clip(115 + 125 * np.sin((x - y) / 190.0), 0, 255)
        img = np.dstack([b, g, r]).astype(np.uint8)
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
    Extracts:
    1. Colour Spectrum -> Pure Melodic Chords + Warm/Bright Red 7th Alert
    2. Pixel Density -> Melodic Rhythm Rate
    3. Visual Shapes -> Clean Synth Filter, Sustain, & Arpeggio Direction
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
    # 1. COLOUR SPECTRUMS -> HARMONIOUS CHORDS + RED/WARM 7TH CHORD ALERT
    # =========================================================================
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_vals = hsv_roi[:, :, 0][mask].astype(np.float32)  # 0..179
    s_vals = hsv_roi[:, :, 1][mask].astype(np.float32) / 255.0
    v_vals = hsv_roi[:, :, 2][mask].astype(np.float32) / 255.0

    chroma_weight = s_vals * np.clip(v_vals * 1.5, 0.05, 1.0)

    # 6 Hue Spectrum Bins:
    # 0: Red/Orange (0-20 & 165-180), 1: Yellow (20-38), 2: Green (38-85),
    # 3: Cyan (85-108), 4: Blue (108-135), 5: Purple (135-165)
    bins = np.zeros(6, dtype=np.float64)
    bins[0] = np.sum(chroma_weight[(h_vals < 20) | (h_vals >= 165)])
    bins[1] = np.sum(chroma_weight[(h_vals >= 20) & (h_vals < 38)])
    bins[2] = np.sum(chroma_weight[(h_vals >= 38) & (h_vals < 85)])
    bins[3] = np.sum(chroma_weight[(h_vals >= 85) & (h_vals < 108)])
    bins[4] = np.sum(chroma_weight[(h_vals >= 108) & (h_vals < 135)])
    bins[5] = np.sum(chroma_weight[(h_vals >= 135) & (h_vals < 165)])

    total_chroma = np.sum(bins)
    if total_chroma > 0.5:
        spectrum = bins / total_chroma
        dom_idx = int(np.argmax(spectrum))
    else:
        spectrum = np.array([0.0, 0.0, 0.1, 0.1, 0.8, 0.0], dtype=np.float64)
        dom_idx = 4

    # Calculate Warm/Bright Red-Orange Alert Level (0.0 to 1.0)
    # High when over red/orange/bright yellow hotspots -> activates the 7th chord voice!
    mean_v = float(np.mean(v_vals))
    warm_ratio = float(spectrum[0] * 1.35 + spectrum[1] * 0.75)
    state.warm_alert_7th = float(np.clip(warm_ratio * (0.5 + 0.8 * mean_v), 0.0, 1.0))

    state.active_chord_idx = dom_idx
    chord_bank = CHORDS_WARM_KEY if state.preset == 0 else CHORDS_DEEP_KEY
    base_name = chord_bank[dom_idx][0]
    if state.warm_alert_7th > 0.35 and dom_idx not in (0, 1):
        state.chord_name = f"{base_name} + Warm 7th Alert"
    else:
        state.chord_name = base_name

    hue_centers = [0.03, 0.14, 0.35, 0.53, 0.65, 0.82]
    state.dominant_hue = hue_centers[dom_idx]
    state.saturation = float(np.clip(np.mean(s_vals) * 1.5, 0.25, 1.0))

    # =========================================================================
    # 2. PIXEL DENSITY -> MELODIC RHYTHM
    # =========================================================================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    stipple = cv2.absdiff(gray, blur)[mask]
    stipple_ratio = float(np.mean(stipple > 10))
    bright_ratio = float(np.mean(v_vals > 0.32))
    variance_norm = float(np.clip(np.std(v_vals) * 3.0, 0.0, 1.0))

    density = float(np.clip(bright_ratio * 0.45 + stipple_ratio * 1.7 + variance_norm * 0.35, 0.0, 1.0))
    state.pixel_density = density
    # Gentle, musical rhythm range: 1.4 Hz (84 BPM) to 7.5 Hz (450 BPM sixteenth notes)
    # Red 7th alert zones get a slight upbeat tempo boost
    alert_boost = 1.0 + state.warm_alert_7th * 0.35
    state.rhythm_hz = (1.4 + (density ** 1.3) * 5.8) * alert_boost

    # =========================================================================
    # 3. VISUAL SHAPES -> CLEAN SYNTHESIZER PARAMETERS
    # =========================================================================
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    jxx = float(np.mean(gx[mask] ** 2)) + 1e-5
    jyy = float(np.mean(gy[mask] ** 2)) + 1e-5
    jxy = float(np.mean(gx[mask] * gy[mask]))

    trace = jxx + jyy
    det_term = np.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * (jxy ** 2)))
    lambda1 = 0.5 * (trace + det_term)
    lambda2 = 0.5 * (trace - det_term)

    elongation = float(np.clip(((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-5)) * 1.4, 0.0, 1.0))
    angle_rad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    shape_angle = float((angle_rad + np.pi / 2.0) / np.pi)

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))[mask]
    jaggedness = float(np.clip(np.mean(lap) / 65.0, 0.0, 1.0))

    state.shape_elongation = elongation
    state.shape_jaggedness = jaggedness
    state.shape_angle = shape_angle

    if state.warm_alert_7th > 0.45:
        state.shape_label = "Warm 7th Beacon (Bright Harmonic Pulse)"
    elif jaggedness > 0.42:
        state.shape_label = "Crisp Contours (Open Filter & Harmonics)"
    elif elongation > 0.50:
        state.shape_label = "Linear Streak (Crisp Pluck Articulation)"
    else:
        state.shape_label = "Smooth Cloud (Warm Mellow Pad)"


def audio_callback(outdata, frames, time_info, status):
    """
    Pure, Pitch-Stable Melodic Synthesizer:
    - Zero pitch-gliding or FM wobble
    - Warm triad pads + Red/Orange 7th Chord Beacon
    - Melodic arpeggiator driven by Pixel Density
    - Minimal, clean room reflection (12% wet)
    """
    a = 0.15
    state.s_density += a * (state.pixel_density - state.s_density)
    state.s_alert_7th += a * (state.warm_alert_7th - state.s_alert_7th)
    state.s_elongation += a * (state.shape_elongation - state.s_elongation)
    state.s_jaggedness += a * (state.shape_jaggedness - state.s_jaggedness)
    state.s_angle += a * (state.shape_angle - state.s_angle)

    # Smooth Hold-to-Play Envelope
    target = 1.0 if (state.is_pressing and not state.menu_open) else 0.0
    rate = 0.0028 if target > 0.5 else 0.00065  # Faster, cleaner release
    t_idx = np.arange(1, frames + 1, dtype=np.float32)
    env = state.pad_env + (target - state.pad_env) * (1.0 - (1.0 - rate) ** t_idx)
    state.pad_env = float(env[-1])

    chord_bank = CHORDS_WARM_KEY if state.preset == 0 else CHORDS_DEEP_KEY
    chord_freqs = chord_bank[state.active_chord_idx][1]  # [Root, 3rd, 5th, Oct, 7th_Alert]

    # =========================================================================
    # MELODIC RHYTHM GENERATOR (Driven by Pixel Density)
    # =========================================================================
    rhythm_inc = state.rhythm_hz / SAMPLE_RATE
    r_phases = state.rhythm_phase + np.arange(1, frames + 1, dtype=np.float64) * rhythm_inc
    triggers = np.floor(r_phases) > np.floor(np.insert(r_phases[:-1], 0, state.rhythm_phase))
    state.rhythm_phase = float(r_phases[-1] % 1.0)
    state.visual_pulse = float(np.exp(-state.rhythm_phase * 4.5)) if state.is_pressing else 0.0

    if np.any(triggers) and state.is_pressing and not state.menu_open:
        slot = state.next_note_slot
        # Include the 7th chord note in the melody whenever Warm/Red Alert is active!
        if state.s_alert_7th > 0.28:
            melody_pool = [chord_freqs[0], chord_freqs[1], chord_freqs[2], chord_freqs[4], chord_freqs[3], chord_freqs[4] * 2.0]
        else:
            melody_pool = [chord_freqs[0], chord_freqs[1], chord_freqs[2], chord_freqs[3], chord_freqs[1] * 2.0]

        # Shape Angle controls whether melody ascends, descends, or alternates
        if state.s_angle > 0.6:
            idx = state.melody_step % len(melody_pool)
        elif state.s_angle < 0.35:
            idx = (len(melody_pool) - 1) - (state.melody_step % len(melody_pool))
        else:
            # Smooth up-and-down wave pattern
            cycle = len(melody_pool) * 2 - 2
            pos = state.melody_step % max(1, cycle)
            idx = pos if pos < len(melody_pool) else cycle - pos

        state.melody_step += 1
        state.note_freqs[slot] = melody_pool[idx]
        state.note_envs[slot] = min(1.0, 0.40 + state.s_density * 0.45 + state.s_alert_7th * 0.15)
        state.note_pans[slot] = float(np.clip(0.5 + 0.32 * np.sin(state.melody_step * 1.5), 0.15, 0.85))
        state.next_note_slot = (slot + 1) % 6

    # =========================================================================
    # WARM HARMONIOUS PAD SYNTHESIS (Zero Wobble, Pure Pitch-Locked Harmonics)
    # =========================================================================
    pad_L = np.zeros(frames, dtype=np.float64)
    pad_R = np.zeros(frames, dtype=np.float64)

    # Voice 4 is the 7th Alert interval: quiet in cool zones, prominent in warm/red zones
    voice_gains = [
        0.28,                           # Root
        0.22,                           # 3rd
        0.22,                           # 5th
        0.15,                           # Octave
        0.38 * state.s_alert_7th,       # 7th Chord Alert Voice (Red/Warm Zone of Interest)
    ]

    # Visual Shape Jaggedness + Warm Alert adds warm upper-octave harmonic brightness
    octave_harmonic_mix = 0.12 + 0.35 * state.s_jaggedness + 0.25 * state.s_alert_7th

    for i in range(5):
        if voice_gains[i] < 0.002:
            continue
        f_hz = chord_freqs[i]
        inc_fund = (2.0 * np.pi * f_hz) / SAMPLE_RATE
        inc_oct = (4.0 * np.pi * f_hz) / SAMPLE_RATE

        p_fund = state.pad_phases[i] + t_idx * inc_fund
        p_oct = state.oct_phases[i] + t_idx * inc_oct
        state.pad_phases[i] = float(p_fund[-1] % (2.0 * np.pi))
        state.oct_phases[i] = float(p_oct[-1] % (2.0 * np.pi))

        # Pure fundamental sine + warm 2nd harmonic (no detune wobble or FM clang)
        wave = np.sin(p_fund) + octave_harmonic_mix * np.sin(p_oct)

        # Gentle rhythmic breathing when in a Red 7th Alert zone
        if i == 4 and state.s_alert_7th > 0.2:
            beacon_pulse = 0.75 + 0.25 * np.sin(2.0 * np.pi * r_phases)
            wave = wave * beacon_pulse

        pan = 0.25 + 0.12 * i
        pad_L += wave * (1.0 - pan) * voice_gains[i]
        pad_R += wave * pan * voice_gains[i]

    pad_L *= env
    pad_R *= env

    # =========================================================================
    # MELODIC PLUCK / ELECTRIC PIANO VOICES (Pure Harmonic Envelopes)
    # =========================================================================
    mel_L = np.zeros(frames, dtype=np.float64)
    mel_R = np.zeros(frames, dtype=np.float64)

    # Shape Elongation controls sustain: round shapes = warm sustain, linear streaks = crisp pluck
    decay_rate = 0.99935 - state.s_elongation * 0.00035 - state.s_density * 0.00025
    decay_curve = decay_rate ** t_idx

    for i in range(6):
        if state.note_envs[i] > 0.001:
            n_env = state.note_envs[i] * decay_curve
            state.note_envs[i] = float(n_env[-1])
            n_inc = (2.0 * np.pi * state.note_freqs[i]) / SAMPLE_RATE
            n_phase = state.note_phases[i] + t_idx * n_inc
            state.note_phases[i] = float(n_phase[-1] % (2.0 * np.pi))

            # Warm Rhodes/Marimba tone: Fundamental + decaying 2nd & 3rd integer harmonics
            overtone_amt = (0.25 + 0.45 * state.s_jaggedness + 0.35 * state.s_alert_7th) * n_env
            tone = (
                np.sin(n_phase)
                + overtone_amt * np.sin(n_phase * 2.0)
                + (overtone_amt * 0.35) * np.sin(n_phase * 3.0)
            ) * n_env * 0.22

            pan = state.note_pans[i]
            mel_L += tone * (1.0 - pan)
            mel_R += tone * pan

    raw_L = (pad_L * 0.26 + mel_L).astype(np.float32)
    raw_R = (pad_R * 0.26 + mel_R).astype(np.float32)

    # Smooth Low-Pass Filter (Opens up on jagged shapes and Red 7th Alert zones)
    cutoff = float(np.clip(0.14 + state.s_jaggedness * 0.40 + state.s_alert_7th * 0.30, 0.10, 0.85))
    filt_L = np.zeros(frames, dtype=np.float32)
    filt_R = np.zeros(frames, dtype=np.float32)
    lp_l, lp_r = state.lp_L, state.lp_R
    for n in range(frames):
        lp_l += cutoff * (raw_L[n] - lp_l)
        lp_r += cutoff * (raw_R[n] - lp_r)
        filt_L[n] = lp_l
        filt_R[n] = lp_r
    state.lp_L, state.lp_R = float(lp_l), float(lp_r)

    # Minimal, Tight Room Reflection (18% feedback, 12% wet mix)
    idx = (state.delay_ptr + np.arange(frames)) % DELAY_SAMPLES
    del_L = state.delay_L[idx]
    del_R = state.delay_R[idx]

    fb = 0.18
    state.delay_L[idx] = filt_L + del_R * fb
    state.delay_R[idx] = filt_R + del_L * fb
    state.delay_ptr = int((state.delay_ptr + frames) % DELAY_SAMPLES)

    outdata[:, 0] = np.tanh(filt_L + del_L * 0.12)
    outdata[:, 1] = np.tanh(filt_R + del_R * 0.12)


def draw_interface(gallery):
    """Renders the full-screen interface, top bar, 7th-alert telemetry, and reactive lens."""
    frame = gallery.centered_canvas.copy()

    # Top Navigation Bar
    cv2.rectangle(frame, (0, 0), (WIN_W, TOP_BAR_H), (22, 24, 30), -1)
    cv2.line(frame, (0, TOP_BAR_H), (WIN_W, TOP_BAR_H), (60, 65, 75), 1)

    menu_col = (95, 150, 245) if state.menu_open else (45, 52, 65)
    cv2.rectangle(frame, (16, 10), (195, 48), menu_col, -1, cv2.LINE_AA)
    cv2.putText(frame, "Images Menu [M]", (28, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (207, 10), (370, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, "+ Add Picture [O]", (218, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 235, 210), 1, cv2.LINE_AA)

    p_name = "Key: C Major Warm [SPACE]" if state.preset == 0 else "Key: F Major Deep [SPACE]"
    p_col = (60, 100, 75) if state.preset == 0 else (85, 65, 105)
    cv2.rectangle(frame, (382, 10), (640, 48), p_col, -1, cv2.LINE_AA)
    cv2.putText(frame, p_name, (394, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (652, 10), (790, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, f"Zoom: {gallery.zoom:.1f}x [R]", (662, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 220, 150), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (802, 10), (960, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, "Full Screen [F]", (816, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (WIN_W - 120, 10), (WIN_W - 16, 48), (60, 45, 155), -1, cv2.LINE_AA)
    cv2.putText(frame, "Quit [Q]", (WIN_W - 98, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if not state.menu_open:
        # Telemetry Panel
        panel_x, panel_y = 22, TOP_BAR_H + 14
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 455, panel_y + 112), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        border_c = (80, 110, 255) if state.s_alert_7th > 0.35 else (70, 75, 90)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 455, panel_y + 112), border_c, 1)

        chord_col = (110, 170, 255) if state.s_alert_7th > 0.35 else (150, 240, 190)
        cv2.putText(
            frame,
            f"1. COLOR -> CHORD: {state.chord_name} (7th Alert: {int(state.s_alert_7th*100)}%)",
            (panel_x + 12, panel_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            chord_col,
            1,
            cv2.LINE_AA,
        )

        bpm = int(state.rhythm_hz * 60)
        cv2.putText(
            frame,
            f"2. DENSITY -> RHYTHM: {state.rhythm_hz:.1f} notes/s ({bpm} BPM)",
            (panel_x + 12, panel_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (250, 220, 140),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"3. SHAPE -> SYNTH: {state.shape_label}",
            (panel_x + 12, panel_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (170, 210, 255),
            1,
            cv2.LINE_AA,
        )

        # Reactive Cursor Lens
        hsv_swatch = np.uint8([[[int(state.dominant_hue * 179), int(state.saturation * 255), 245]]])
        ring_bgr = tuple(int(c) for c in cv2.cvtColor(hsv_swatch, cv2.COLOR_HSV2BGR)[0, 0])

        rhythm_expand = int(state.visual_pulse * 10)
        cv2.circle(
            frame,
            (state.cursor_x, state.cursor_y),
            state.lens_radius + rhythm_expand,
            ring_bgr,
            2,
            cv2.LINE_AA,
        )

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
            f"Map: {title} | Scroll or [+/-] Zoom | Right-Drag or [Arrows] Pan | [F] Fullscreen | [W/S] Lens | [Q] Quit",
            (20, WIN_H - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (165, 170, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        # Image Selection Menu Overlay
        overlay = frame.copy()
        mx1, my1, mx2, my2 = (WIN_W - 980) // 2, 95, (WIN_W + 980) // 2, WIN_H - 75
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (90, 120, 180), 2)

        cv2.putText(
            frame,
            "Select an Image to Sonify (Click card or press 1-9, or press [O] to browse files)",
            (mx1 + 35, my1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )

        cols, card_w, card_h = 3, 280, 210
        start_x, start_y, gap_x, gap_y = mx1 + 35, my1 + 70, 35, 35

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


def toggle_fullscreen(win_name):
    state.fullscreen = not state.fullscreen
    prop = cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)


def make_mouse_handler(gallery, win_name):
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
                elif 382 <= x <= 640:
                    state.preset = 1 - state.preset
                elif 652 <= x <= 790:
                    gallery.reset_view()
                elif 802 <= x <= 960:
                    toggle_fullscreen(win_name)
                elif x >= WIN_W - 120:
                    state.running = False
                return

            if state.menu_open:
                mx1, my1 = (WIN_W - 980) // 2, 95
                cols, card_w, card_h = 3, 280, 210
                start_x, start_y, gap_x, gap_y = mx1 + 35, my1 + 70, 35, 35
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
    win_name = "Melodic Earth Sonifier"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_name, make_mouse_handler(gallery, win_name))

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
            elif low_key in (ord("f"), ord("F")):
                toggle_fullscreen(win_name)
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