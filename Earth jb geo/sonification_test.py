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
DELAY_SAMPLES = int(SAMPLE_RATE * 0.46)  # 460ms Eno tape-loop delay

# Brian Eno Harmonic Scales (Hz)
# Preset 0: "Music for Airports / Apollo" (Db Lydian / Major 9th)
SCALE_ENO_LIGHT = np.array([
    138.59, 155.56, 174.61, 196.00, 207.65, 233.08, 261.63,
    277.18, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25, 622.25, 698.46
], dtype=np.float64)

# Preset 1: "Dark Apollo / Deep Space" (C Dorian / Minor 11th)
SCALE_ENO_DARK = np.array([
    98.00, 110.00, 116.54, 130.81, 146.83, 155.56, 174.61,
    196.00, 220.00, 233.08, 261.63, 293.66, 311.13, 349.23, 392.00, 466.16
], dtype=np.float64)


class EnoSonifierState:
    def __init__(self):
        self.running = True
        self.menu_open = False
        self.is_pressing = False
        self.preset = 0  # 0 = Airports/Lydian, 1 = Dark Apollo/Dorian
        self.lens_radius = 46

        self.cursor_x = WIN_W // 2
        self.cursor_y = (WIN_H + TOP_BAR_H) // 2
        self.velocity = 0.0

        # Extracted Color & Spatial Metrics (0.0 to 1.0)
        self.hue = 0.35
        self.sat = 0.5
        self.lum = 0.2
        self.edge = 0.1

        # Distinct Biome Color Weights (Green, Purple, Orange/Red, Cyan/Blue)
        self.w_green = 0.0
        self.w_purple = 0.0
        self.w_orange = 0.0
        self.w_cyan = 0.0
        self.dominant_label = "Deep Space / Ocean"

        # Smoothed Synthesis Parameters
        self.s_hue = 0.35
        self.s_sat = 0.5
        self.s_lum = 0.2
        self.s_edge = 0.1
        self.s_green = 0.0
        self.s_purple = 0.0
        self.s_orange = 0.0
        self.s_cyan = 0.0

        # Audio Envelopes & Phase Accumulators
        self.pad_env = 0.0
        self.phases = np.zeros(8, dtype=np.float64)
        self.lfo_phase = 0.0

        # Generative Tape-Loop Chime Voices (3 polyphonic chime slots)
        self.chime_freqs = np.array([277.18, 349.23, 415.30], dtype=np.float64)
        self.chime_envs = np.zeros(3, dtype=np.float64)
        self.chime_phases = np.zeros(3, dtype=np.float64)
        self.next_chime_slot = 0
        self.last_scale_idx = -1
        self.chime_cooldown = 0

        # Stereo Ping-Pong Delay Buffer
        self.delay_L = np.zeros(DELAY_SAMPLES, dtype=np.float32)
        self.delay_R = np.zeros(DELAY_SAMPLES, dtype=np.float32)
        self.delay_ptr = 0
        self.lp_L = 0.0
        self.lp_R = 0.0


state = EnoSonifierState()


def generate_aerosol_fallback():
    """Creates a synthetic high-contrast aerosol map if no local image is found yet."""
    h, w = 720, 1280
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), (22, 26, 22), dtype=np.uint8)

    # Green aerosol plumes
    green_wave = np.sin(x / 110.0 + np.cos(y / 90.0) * 2.0) * np.cos(y / 140.0)
    g_mask = np.clip((green_wave - 0.15) * 2.2, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] + g_mask * 220, 0, 255).astype(np.uint8)

    # Saharan purple dust band
    purple_band = np.exp(-((y - h * 0.42 + np.sin(x / 130.0) * 45.0) ** 2) / (2 * 55.0**2))
    p_mask = purple_band * np.clip(np.sin(x / 260.0), 0, 1)
    img[:, :, 0] = np.clip(img[:, :, 0] + p_mask * 210, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + p_mask * 120, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] + p_mask * 200, 0, 255).astype(np.uint8)

    # Central African orange/red smoke core
    smoke = np.exp(-((x - w * 0.54) ** 2 + (y - h * 0.58) ** 2) / (2 * 65.0**2))
    img[:, :, 2] = np.clip(img[:, :, 2] + smoke * 245, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + smoke * 95, 0, 255).astype(np.uint8)

    # Southern ocean cyan swirls
    cyan_swirl = np.exp(-((y - h * 0.78) ** 2) / (2 * 60.0**2)) * (0.5 + 0.5 * np.sin(x / 55.0))
    img[:, :, 0] = np.clip(img[:, :, 0] + cyan_swirl * 185, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + cyan_swirl * 145, 0, 255).astype(np.uint8)

    return img


class ImageGallery:
    def __init__(self):
        self.entries = []  # List of (title, image_bgr)
        self.current_idx = 0
        self.centered_canvas = None
        self.img_rect = (0, TOP_BAR_H, WIN_W, WIN_H - TOP_BAR_H)  # (x, y, w, h)
        self.load_initial_images()

    def load_initial_images(self):
        target_name = "Aerosols_2025-08-19T20_00_00_5760x3240.jpg"
        loaded_paths = set()

        # 1. Prioritize the user's NASA Aerosol image if present
        if os.path.exists(target_name):
            img = cv2.imread(target_name)
            if img is not None:
                self.entries.append(("NASA Global Aerosols", img))
                loaded_paths.add(os.path.abspath(target_name))

        # 2. Scan folder for any other images (.jpg, .jpeg, .png, .webp, .bmp)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for path in sorted(glob.glob(ext)):
                abs_p = os.path.abspath(path)
                if abs_p not in loaded_paths:
                    img = cv2.imread(path)
                    if img is not None:
                        label = os.path.splitext(os.path.basename(path))[0][:28]
                        self.entries.append((label, img))
                        loaded_paths.add(abs_p)

        # 3. Always include built-in NASA Aerosol fallback & Biomes so menu has multiple choices
        if not self.entries:
            self.entries.append(("NASA Global Aerosols (Built-in)", generate_aerosol_fallback()))
        self.entries.append(("Synthetic Nebula & Reefs", self._make_nebula_map()))

        self.select_image(0)

    def _make_nebula_map(self):
        h, w = 720, 1280
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        b = np.clip(128 + 120 * np.sin(x / 140.0 + y / 200.0), 0, 255)
        g = np.clip(110 + 110 * np.cos(x / 95.0 - y / 130.0), 0, 255)
        r = np.clip(120 + 125 * np.sin((x - y) / 160.0), 0, 255)
        return np.dstack([b, g, r]).astype(np.uint8)

    def select_image(self, idx):
        self.current_idx = idx % len(self.entries)
        _, raw_img = self.entries[self.current_idx]

        # Center the image inside the main viewport while preserving aspect ratio
        avail_w = WIN_W - 40
        avail_h = WIN_H - TOP_BAR_H - 40
        ih, iw = raw_img.shape[:2]
        scale = min(avail_w / iw, avail_h / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = cv2.resize(raw_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((WIN_H, WIN_W, 3), (14, 15, 18), dtype=np.uint8)

        x_off = (WIN_W - new_w) // 2
        y_off = TOP_BAR_H + (WIN_H - TOP_BAR_H - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        # Subtle frame border around centered image
        cv2.rectangle(
            canvas,
            (x_off - 1, y_off - 1),
            (x_off + new_w + 1, y_off + new_h + 1),
            (65, 70, 80),
            1,
        )
        self.centered_canvas = canvas
        self.img_rect = (x_off, y_off, new_w, new_h)

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
    """Extracts high-variance Hue, Saturation, Biome Weights, and Edge Density."""
    ix, iy, iw, ih = img_rect
    # Clamp sampling coordinate to the centered image bounds
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

    bgr_pixels = roi[mask].astype(np.float32) / 255.0
    b, g, r = bgr_pixels[:, 0], bgr_pixels[:, 1], bgr_pixels[:, 2]
    lum_arr = 0.114 * b + 0.587 * g + 0.299 * r

    # Focus color detection on the brightest 40% of pixels in the lens so dark backgrounds don't mute color
    bright_thresh = np.percentile(lum_arr, 55)
    active = lum_arr >= max(0.04, bright_thresh)
    if np.any(active):
        mr, mg, mb = float(np.mean(r[active])), float(np.mean(g[active])), float(np.mean(b[active]))
    else:
        mr, mg, mb = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))

    # Convert mean active RGB to HSV (0.0 - 1.0)
    cmax, cmin = max(mr, mg, mb), min(mr, mg, mb)
    delta = cmax - cmin
    sat = (delta / cmax) if cmax > 0.02 else 0.0

    if delta < 1e-4:
        hue = 0.0
    elif cmax == mr:
        hue = ((mg - mb) / delta) % 6.0 / 6.0
    elif cmax == mg:
        hue = (((mb - mr) / delta) + 2.0) / 6.0
    else:
        hue = (((mr - mg) / delta) + 4.0) / 6.0

    # Calculate distinct color zone weights (amplified for high musical variance)
    w_green = np.clip((mg - max(mr, mb) * 0.75) * 3.2, 0.0, 1.0)
    w_purple = np.clip((min(mr, mb) - mg * 0.6) * 3.5 + (mr * mb * 1.8), 0.0, 1.0)
    w_orange = np.clip((mr - mb * 1.1 - mg * 0.35) * 3.2, 0.0, 1.0)
    w_cyan = np.clip((min(mb, mg) - mr * 0.7) * 3.2, 0.0, 1.0)

    # Identify active color biome for HUD feedback
    weights = {
        "Green Sulfate Plume (Chorus Pad)": w_green,
        "Purple Saharan Dust (Glass 9ths)": w_purple,
        "Orange/Red Smoke (Warm DX7 FM)": w_orange,
        "Cyan Sea Salt Swirl (Octave Shimmer)": w_cyan,
    }
    best_label, best_val = max(weights.items(), key=lambda kv: kv[1])
    state.dominant_label = best_label if best_val > 0.14 else "Dark Space / Sub Pedal"

    # Edge density via Sobel gradients
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(np.clip(np.mean(np.hypot(gx, gy)[mask]) / 95.0, 0.0, 1.0))

    state.hue = float(hue)
    state.sat = float(np.clip(sat * 1.5, 0.0, 1.0))
    state.lum = float(np.clip(cmax * 1.25, 0.0, 1.0))
    state.edge = edge_density
    state.w_green = float(w_green)
    state.w_purple = float(w_purple)
    state.w_orange = float(w_orange)
    state.w_cyan = float(w_cyan)

    # Trigger generative Eno chime when crossing into a new harmonic step
    scale = SCALE_ENO_LIGHT if state.preset == 0 else SCALE_ENO_DARK
    color_pos = (
        state.hue * 0.55
        + state.w_green * 0.2
        + state.w_purple * 0.35
        + state.w_orange * 0.5
        + state.lum * 0.25
    ) % 1.0
    scale_idx = int(np.clip(color_pos * len(scale), 0, len(scale) - 1))

    if state.chime_cooldown > 0:
        state.chime_cooldown -= 1

    if (scale_idx != state.last_scale_idx or state.edge > 0.45) and state.chime_cooldown == 0:
        slot = state.next_chime_slot
        octave_mul = 2.0 if (state.w_purple > 0.25 or state.w_cyan > 0.25) else 1.0
        state.chime_freqs[slot] = scale[scale_idx] * octave_mul
        state.chime_envs[slot] = min(1.0, 0.45 + state.sat * 0.5)
        state.next_chime_slot = (slot + 1) % 3
        state.last_scale_idx = scale_idx
        state.chime_cooldown = max(3, int(11 - state.edge * 8))


def advance_phase(idx, freq_hz, frames):
    inc = (2.0 * np.pi * freq_hz) / SAMPLE_RATE
    phases = state.phases[idx] + np.arange(1, frames + 1, dtype=np.float64) * inc
    state.phases[idx] = phases[-1] % (2.0 * np.pi)
    return phases


def audio_callback(outdata, frames, time_info, status):
    """Brian Eno Generative Ambient Synthesizer + Stereo Ping-Pong Shimmer Delay."""
    a = 0.12
    state.s_hue += a * (state.hue - state.s_hue)
    state.s_sat += a * (state.sat - state.s_sat)
    state.s_lum += a * (state.lum - state.s_lum)
    state.s_edge += a * (state.edge - state.s_edge)
    state.s_green += a * (state.w_green - state.s_green)
    state.s_purple += a * (state.w_purple - state.s_purple)
    state.s_orange += a * (state.w_orange - state.s_orange)
    state.s_cyan += a * (state.w_cyan - state.s_cyan)

    # Smooth pad envelope (hold trackpad to swell, release for lush fade)
    target = 1.0 if (state.is_pressing and not state.menu_open) else 0.0
    rate = 0.0018 if target > 0.5 else 0.00035
    t_idx = np.arange(1, frames + 1, dtype=np.float32)
    env = state.pad_env + (target - state.pad_env) * (1.0 - (1.0 - rate) ** t_idx)
    state.pad_env = float(env[-1])

    scale = SCALE_ENO_LIGHT if state.preset == 0 else SCALE_ENO_DARK
    n_notes = len(scale)

    # Quantize root and harmony voices based on dominant color zone so colors shift notes clearly
    color_index = int(
        np.clip(
            (state.s_hue * 8.0 + state.s_green * 2.0 + state.s_purple * 4.0 + state.s_orange * 6.0)
            % (n_notes - 6),
            0,
            n_notes - 7,
        )
    )
    f_bass = scale[color_index % 5] * 0.5  # Deep pedal bass
    f_root = scale[color_index]
    f_third = scale[color_index + 2]
    f_fifth = scale[color_index + 4]
    f_ninth = scale[color_index + 6]

    # Slow organic tape-wow LFO
    lfo_inc = (2.0 * np.pi * (0.35 + state.s_edge * 1.8)) / SAMPLE_RATE
    lfo_p = state.lfo_phase + np.arange(1, frames + 1) * lfo_inc
    state.lfo_phase = lfo_p[-1] % (2.0 * np.pi)
    lfo = np.sin(lfo_p)

    # 1. Sub Pedal Drone (Always grounding dark areas)
    p_bass = advance_phase(0, f_bass, frames)
    voice_bass = np.sin(p_bass) * (0.45 - 0.2 * state.s_sat)

    # 2. Green Plume Voice: Lush Detuned Chorus Pad (Root + 3rd)
    p_root1 = advance_phase(1, f_root * 0.997, frames)
    p_root2 = advance_phase(2, f_root * 1.003, frames)
    p_third = advance_phase(3, f_third, frames)
    green_gain = 0.15 + 0.75 * state.s_green
    voice_green = (np.sin(p_root1) + np.sin(p_root2) + 0.8 * np.sin(p_third + 0.4 * np.sin(p_root1))) * 0.35 * green_gain

    # 3. Orange/Red Smoke Voice: Warm DX7 FM Harmonic Swell
    p_fm_mod = advance_phase(4, f_root * 2.0, frames)
    p_fm_car = advance_phase(5, f_fifth, frames)
    fm_index = 0.5 + 3.8 * state.s_orange + 1.2 * state.s_edge
    voice_orange = np.sin(p_fm_car + fm_index * np.sin(p_fm_mod)) * (0.1 + 0.75 * state.s_orange)

    # 4. Purple Dust & Cyan Swirl Voice: High Glass 9ths & Octave Shimmer
    p_ninth = advance_phase(6, f_ninth + lfo * 0.8, frames)
    p_shimmer = advance_phase(7, f_third * 2.0, frames)
    voice_purple = np.sin(p_ninth + 0.6 * np.sin(p_shimmer)) * (0.1 + 0.8 * state.s_purple)
    voice_cyan = np.sin(p_shimmer) * (0.08 + 0.75 * state.s_cyan + 0.3 * state.s_edge)

    # Combine Stereo Pad Voices
    pad_L = (voice_bass + voice_green + voice_orange * 0.8 + voice_purple * 0.4 + voice_cyan * 0.9) * env
    pad_R = (voice_bass + voice_green * 0.85 + voice_orange * 0.5 + voice_purple * 0.95 + voice_cyan * 0.4) * env

    # 5. Asynchronous Eno Tape-Loop Chimes (Ring out even when cursor stops)
    chime_L = np.zeros(frames, dtype=np.float32)
    chime_R = np.zeros(frames, dtype=np.float32)
    decay = 0.99955 ** np.arange(1, frames + 1, dtype=np.float32)

    for i in range(3):
        if state.chime_envs[i] > 0.001:
            c_env = state.chime_envs[i] * decay
            state.chime_envs[i] = float(c_env[-1])
            c_inc = (2.0 * np.pi * state.chime_freqs[i]) / SAMPLE_RATE
            c_p = state.chime_phases[i] + np.arange(1, frames + 1) * c_inc
            state.chime_phases[i] = c_p[-1] % (2.0 * np.pi)
            # DX7 electric piano bell tone (carrier + 3x harmonic modulator decaying faster)
            bell = np.sin(c_p + (c_env * 1.8) * np.sin(c_p * 3.0)) * c_env * 0.22
            if i % 2 == 0:
                chime_L += bell
                chime_R += bell * 0.5
            else:
                chime_L += bell * 0.5
                chime_R += bell

    dry_L = (pad_L * 0.26 + chime_L).astype(np.float32)
    dry_R = (pad_R * 0.26 + chime_R).astype(np.float32)

    # 6. Vectorized Stereo Ping-Pong Delay Buffer
    idx = (state.delay_ptr + np.arange(frames)) % DELAY_SAMPLES
    del_L = state.delay_L[idx]
    del_R = state.delay_R[idx]

    # Warm tape feedback write
    fb = 0.64
    state.delay_L[idx] = np.tanh(dry_L + del_R * fb)
    state.delay_R[idx] = np.tanh(dry_R + del_L * fb)
    state.delay_ptr = int((state.delay_ptr + frames) % DELAY_SAMPLES)

    outdata[:, 0] = np.tanh(dry_L + del_L * 0.55)
    outdata[:, 1] = np.tanh(dry_R + del_R * 0.55)


def draw_interface(gallery):
    """Renders the centered image, top control bar, reactive lens, and selection menu."""
    frame = gallery.centered_canvas.copy()

    # Draw Top Navigation Bar
    cv2.rectangle(frame, (0, 0), (WIN_W, TOP_BAR_H), (22, 24, 30), -1)
    cv2.line(frame, (0, TOP_BAR_H), (WIN_W, TOP_BAR_H), (60, 65, 75), 1)

    # Button 1: Image Menu (M)
    menu_col = (95, 150, 245) if state.menu_open else (45, 52, 65)
    cv2.rectangle(frame, (16, 10), (215, 48), menu_col, -1, cv2.LINE_AA)
    cv2.putText(frame, "Images Menu [M]", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)

    # Button 2: Add File (O)
    cv2.rectangle(frame, (228, 10), (405, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, "+ Add Picture [O]", (242, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 235, 210), 1, cv2.LINE_AA)

    # Button 3: Eno Preset Toggle (SPACE)
    p_name = "Preset: Airports Lydian [SPACE]" if state.preset == 0 else "Preset: Dark Apollo [SPACE]"
    p_col = (70, 95, 65) if state.preset == 0 else (85, 55, 95)
    cv2.rectangle(frame, (418, 10), (725, 48), p_col, -1, cv2.LINE_AA)
    cv2.putText(frame, p_name, (432, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 1, cv2.LINE_AA)

    # Active Biome & Voice Readout
    cv2.putText(frame, f"Voice: {state.dominant_label}", (745, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 230, 250), 1, cv2.LINE_AA)

    # Button 4: Quit Button (Q)
    cv2.rectangle(frame, (WIN_W - 135, 10), (WIN_W - 16, 48), (60, 45, 155), -1, cv2.LINE_AA)
    cv2.putText(frame, "Quit [Q]", (WIN_W - 108, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    if not state.menu_open:
        # Draw reactive sampling lens colored by the detected hue/biome
        hsv_swatch = np.uint8([[[int(state.s_hue * 179), int(max(90, state.s_sat * 255)), 245]]])
        ring_bgr = tuple(int(c) for c in cv2.cvtColor(hsv_swatch, cv2.COLOR_HSV2BGR)[0, 0])
        pulse = int(state.pad_env * 10)
        cv2.circle(frame, (state.cursor_x, state.cursor_y), state.lens_radius + pulse, ring_bgr, 2, cv2.LINE_AA)
        inner_r = int(max(4, state.lens_radius * state.s_edge))
        cv2.circle(frame, (state.cursor_x, state.cursor_y), inner_r, (235, 235, 235), 1, cv2.LINE_AA)

        # Bottom Status Bar
        title = gallery.entries[gallery.current_idx][0]
        cv2.putText(
            frame,
            f"Map: {title}  |  Hold & Drag Trackpad to Play  |  [W/S] Lens Size  |  [M] Switch Image  |  [Q] Quit",
            (20, WIN_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (165, 170, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        # Draw Centered Image Selection Menu Overlay
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

        # Render up to 6 image cards in a 3x2 grid
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

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Top Bar UI Buttons first
            if y <= TOP_BAR_H:
                if 16 <= x <= 215:
                    state.menu_open = not state.menu_open
                elif 228 <= x <= 405:
                    gallery.add_image_via_dialog()
                elif 418 <= x <= 725:
                    state.preset = 1 - state.preset
                elif x >= WIN_W - 135:
                    state.running = False
                return

            # If Selection Menu is open, check thumbnail clicks
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

        elif event == cv2.EVENT_MOUSEMOVE and state.is_pressing and not state.menu_open:
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

            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("m"), ord("M")):
                state.menu_open = not state.menu_open
            elif key in (ord("o"), ord("O")):
                gallery.add_image_via_dialog()
            elif key == ord(" "):
                state.preset = 1 - state.preset
            elif key in (ord("w"), ord("W")):
                state.lens_radius = min(130, state.lens_radius + 6)
            elif key in (ord("s"), ord("S")):
                state.lens_radius = max(14, state.lens_radius - 6)
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(gallery.entries):
                    gallery.select_image(idx)
                    state.menu_open = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()