import os
import sys
import glob
import ctypes
import threading
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
CHORDS_WARM_KEY = [
    ("G7 Alert (Red Hotspot)", np.array([196.00, 246.94, 293.66, 392.00, 349.23], dtype=np.float64)),
    ("Dm7 (Amber Ridge)", np.array([146.83, 174.61, 220.00, 293.66, 261.63], dtype=np.float64)),
    ("F Major (Green Plume)", np.array([174.61, 220.00, 261.63, 349.23, 329.63], dtype=np.float64)),
    ("A Minor (Cyan Swirl)", np.array([110.00, 164.81, 220.00, 261.63, 329.63], dtype=np.float64)),
    ("C Major (Deep Ocean)", np.array([130.81, 164.81, 196.00, 261.63, 246.94], dtype=np.float64)),
    ("G Major (Purple Dust)", np.array([146.83, 196.00, 246.94, 293.66, 329.63], dtype=np.float64)),
]

CHORDS_DEEP_KEY = [
    ("C7 Alert (Red Hotspot)", np.array([130.81, 164.81, 196.00, 261.63, 233.08], dtype=np.float64)),
    ("Gm7 (Amber Ridge)", np.array([98.00, 116.54, 146.83, 196.00, 174.61], dtype=np.float64)),
    ("Bb Major (Green Plume)", np.array([116.54, 146.83, 174.61, 233.08, 220.00], dtype=np.float64)),
    ("D Minor (Cyan Swirl)", np.array([73.42, 110.00, 146.83, 174.61, 220.00], dtype=np.float64)),
    ("F Major (Deep Ocean)", np.array([87.31, 110.00, 130.81, 174.61, 164.81], dtype=np.float64)),
    ("C Major (Purple Dust)", np.array([98.00, 130.81, 164.81, 196.00, 220.00], dtype=np.float64)),
]


class SonifierState:
    def __init__(self):
        self.running = True
        self.fullscreen = True
        self.menu_open = False
        self.is_pressing = False
        self.preset = 0
        self.lens_radius = 48

        # Unified PC + Image Viewer Cursor Position (Always identical)
        self.cursor_x = WIN_W // 2
        self.cursor_y = (WIN_H + TOP_BAR_H) // 2
        self.abs_trackpad_active = False
        self.needs_lens_update = False

        # 1. COLOUR SPECTRUMS -> HARMONIOUS CHORDS & WARM 7TH ALERT
        self.active_chord_idx = 4
        self.chord_name = "C Major (Deep Ocean)"
        self.warm_alert_7th = 0.0
        self.dominant_hue = 0.60
        self.saturation = 0.6

        # 2. PIXEL DENSITY -> MELODIC RHYTHM
        self.pixel_density = 0.15
        self.rhythm_hz = 2.0
        self.rhythm_phase = 0.0
        self.visual_pulse = 0.0

        # 3. VISUAL SHAPES -> CLEAN SYNTH PARAMETERS
        self.shape_elongation = 0.2
        self.shape_jaggedness = 0.15
        self.shape_angle = 0.5
        self.shape_label = "Smooth Cloud (Warm Mellow Pad)"

        # Smoothed Audio Thread Parameters
        self.s_density = 0.15
        self.s_alert_7th = 0.0
        self.s_elongation = 0.2
        self.s_jaggedness = 0.15
        self.s_angle = 0.5
        self.pad_env = 0.0

        # Pitch-Locked Oscillator Phases
        self.pad_phases = np.zeros(5, dtype=np.float64)
        self.oct_phases = np.zeros(5, dtype=np.float64)

        # Clean Melodic Note Voices (6 polyphonic slots)
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


# =============================================================================
# OS-LEVEL PC CURSOR LOCKING & ABSOLUTE TRACKPAD HARDWARE DIGITIZER
# =============================================================================
if sys.platform == "darwin":
    class _CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    _app_services = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
    _app_services.CGWarpMouseCursorPosition.argtypes = [_CGPoint]
    _app_services.CGWarpMouseCursorPosition.restype = ctypes.c_int32
    _app_services.CGAssociateMouseAndMouseCursorPosition.argtypes = [ctypes.c_bool]
    _app_services.CGAssociateMouseAndMouseCursorPosition.restype = ctypes.c_int32

    def warp_os_cursor(x, y):
        """Physically moves the macOS system cursor to (x, y)."""
        _app_services.CGWarpMouseCursorPosition(_CGPoint(float(x), float(y)))
        _app_services.CGAssociateMouseAndMouseCursorPosition(True)

    def clip_os_cursor_to_rect(img_rect):
        pass  # Handled continuously by clamping + CGWarpMouseCursorPosition

    def release_os_cursor_clip():
        _app_services.CGAssociateMouseAndMouseCursorPosition(True)

elif sys.platform == "win32":
    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    _user32 = ctypes.windll.user32

    def warp_os_cursor(x, y):
        """Physically moves the Windows system cursor to (x, y)."""
        _user32.SetCursorPos(int(x), int(y))

    def clip_os_cursor_to_rect(img_rect):
        """Physically confines the Windows PC cursor inside the picture rectangle."""
        ix, iy, iw, ih = img_rect
        rect = _RECT(int(ix), int(iy), int(ix + iw), int(iy + ih))
        _user32.ClipCursor(ctypes.byref(rect))

    def release_os_cursor_clip():
        _user32.ClipCursor(None)

else:
    def warp_os_cursor(x, y):
        pass

    def clip_os_cursor_to_rect(img_rect):
        pass

    def release_os_cursor_clip():
        pass


def teleport_cursor_from_trackpad_uv(u_raw, v_raw, gallery):
    """
    Maps normalized physical trackpad touch coordinates (u_raw, v_raw in 0.0..1.0)
    directly to the picture's pixel corners and teleports the PC cursor there.
    - Top-Right of Trackpad (u=1, v=0) -> Top-Right of Picture
    - Bottom-Left of Trackpad (u=0, v=1) -> Bottom-Left of Picture
    """
    ix, iy, iw, ih = gallery.img_rect
    # Trim 4% physical bezel margin so touching the corner of the trackpad hits the exact corner pixel
    u = float(np.clip((u_raw - 0.04) / 0.92, 0.0, 1.0))
    v = float(np.clip((v_raw - 0.04) / 0.92, 0.0, 1.0))

    pic_x = int(ix + u * max(1, iw - 1))
    pic_y = int(iy + v * max(1, ih - 1))

    state.cursor_x = pic_x
    state.cursor_y = pic_y
    warp_os_cursor(pic_x, pic_y)
    state.needs_lens_update = True


def start_absolute_trackpad_listener(gallery):
    """
    Starts a background hardware listener that reads absolute (0.0 - 1.0) touch positions
    directly from the physical trackpad glass on macOS (MultitouchSupport) or Windows (Raw Input HID).
    """
    if sys.platform == "darwin":
        try:
            mt = ctypes.CDLL("/System/Library/PrivateFrameworks/MultitouchSupport.framework/MultitouchSupport")
            cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

            class MTPoint(ctypes.Structure):
                _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

            class MTReadout(ctypes.Structure):
                _fields_ = [("pos", MTPoint), ("vel", MTPoint)]

            class MTTouch(ctypes.Structure):
                _fields_ = [
                    ("frame", ctypes.c_int),
                    ("timestamp", ctypes.c_double),
                    ("identifier", ctypes.c_int),
                    ("state", ctypes.c_int),
                    ("finger_id", ctypes.c_int),
                    ("hand_id", ctypes.c_int),
                    ("normalized", MTReadout),
                    ("size", ctypes.c_float),
                    ("zero1", ctypes.c_int),
                    ("angle", ctypes.c_float),
                    ("major_axis", ctypes.c_float),
                    ("minor_axis", ctypes.c_float),
                    ("mm", MTReadout),
                    ("zero2", ctypes.c_int * 2),
                    ("unk2", ctypes.c_float),
                ]

            MTContactCallback = ctypes.CFUNCTYPE(
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(MTTouch),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
            )

            def _mt_callback(device, data_ptr, n_fingers, timestamp, frame):
                if not state.running:
                    return 0
                if n_fingers > 0 and bool(data_ptr):
                    touch = data_ptr[0]
                    # States 3, 4, 5 indicate active finger contact on the trackpad surface
                    if touch.state in (3, 4, 5):
                        u = float(touch.normalized.pos.x)          # 0.0 (Left) to 1.0 (Right)
                        v = float(1.0 - touch.normalized.pos.y)    # 0.0 (Top) to 1.0 (Bottom)
                        teleport_cursor_from_trackpad_uv(u, v, gallery)
                        if not state.menu_open:
                            state.is_pressing = True
                    else:
                        state.is_pressing = False
                else:
                    state.is_pressing = False
                return 0

            # Keep reference alive for the lifetime of the program
            state._mt_cb_ref = MTContactCallback(_mt_callback)

            mt.MTDeviceCreateList.restype = ctypes.c_void_p
            cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
            cf.CFArrayGetCount.restype = ctypes.c_long
            cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
            cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
            mt.MTRegisterContactFrameCallback.argtypes = [ctypes.c_void_p, MTContactCallback]
            mt.MTDeviceStart.argtypes = [ctypes.c_void_p, ctypes.c_int]

            dev_list = mt.MTDeviceCreateList()
            if dev_list:
                count = cf.CFArrayGetCount(dev_list)
                for i in range(count):
                    dev = cf.CFArrayGetValueAtIndex(dev_list, i)
                    mt.MTRegisterContactFrameCallback(dev, state._mt_cb_ref)
                    mt.MTDeviceStart(dev, 0)
                if count > 0:
                    state.abs_trackpad_active = True
        except Exception as e:
            print(f"macOS absolute trackpad fallback to clamped cursor: {e}")

    elif sys.platform == "win32":
        # Windows Precision Touchpad Raw Input HID listener thread
        def _win_touchpad_thread():
            try:
                user32 = ctypes.windll.user32
                hid = ctypes.windll.hid

                class RAWINPUTDEVICE(ctypes.Structure):
                    _fields_ = [
                        ("usUsagePage", ctypes.c_ushort),
                        ("usUsage", ctypes.c_ushort),
                        ("dwFlags", ctypes.c_ulong),
                        ("hwndTarget", ctypes.c_void_p),
                    ]

                class RAWINPUTHEADER(ctypes.Structure):
                    _fields_ = [
                        ("dwType", ctypes.c_ulong),
                        ("dwSize", ctypes.c_ulong),
                        ("hDevice", ctypes.c_void_p),
                        ("wParam", ctypes.c_size_t),
                    ]

                class RAWINPUTHID(ctypes.Structure):
                    _fields_ = [
                        ("dwSizeHid", ctypes.c_ulong),
                        ("dwCount", ctypes.c_ulong),
                        ("bRawData", ctypes.c_ubyte * 1),
                    ]

                class RAWINPUT(ctypes.Structure):
                    _fields_ = [("header", RAWINPUTHEADER), ("hid", RAWINPUTHID)]

                class HIDP_VALUE_CAPS(ctypes.Structure):
                    _fields_ = [
                        ("UsagePage", ctypes.c_ushort),
                        ("ReportID", ctypes.c_ubyte),
                        ("IsAlias", ctypes.c_ubyte),
                        ("BitField", ctypes.c_ushort),
                        ("LinkCollection", ctypes.c_ushort),
                        ("LinkUsage", ctypes.c_ushort),
                        ("LinkUsagePage", ctypes.c_ushort),
                        ("IsRange", ctypes.c_ubyte),
                        ("IsStringRange", ctypes.c_ubyte),
                        ("IsDesignatorRange", ctypes.c_ubyte),
                        ("IsAbsolute", ctypes.c_ubyte),
                        ("HasNull", ctypes.c_ubyte),
                        ("Reserved", ctypes.c_ubyte),
                        ("BitSize", ctypes.c_ushort),
                        ("ReportCount", ctypes.c_ushort),
                        ("Reserved2", ctypes.c_ushort * 5),
                        ("UnitsExp", ctypes.c_ulong),
                        ("Units", ctypes.c_ulong),
                        ("LogicalMin", ctypes.c_long),
                        ("LogicalMax", ctypes.c_long),
                        ("PhysicalMin", ctypes.c_long),
                        ("PhysicalMax", ctypes.c_long),
                        ("UsageMin", ctypes.c_ushort),
                        ("UsageMax", ctypes.c_ushort),
                        ("StringMin", ctypes.c_ushort),
                        ("StringMax", ctypes.c_ushort),
                        ("DesignatorMin", ctypes.c_ushort),
                        ("DesignatorMax", ctypes.c_ushort),
                        ("DataIndexMin", ctypes.c_ushort),
                        ("DataIndexMax", ctypes.c_ushort),
                    ]

                class MSG(ctypes.Structure):
                    _fields_ = [
                        ("hwnd", ctypes.c_void_p),
                        ("message", ctypes.c_uint),
                        ("wParam", ctypes.c_size_t),
                        ("lParam", ctypes.c_size_t),
                        ("time", ctypes.c_ulong),
                        ("pt_x", ctypes.c_long),
                        ("pt_y", ctypes.c_long),
                    ]

                HWND_MESSAGE = ctypes.c_void_p(-3)
                hwnd = user32.CreateWindowExW(0, "STATIC", "TPAbsSink", 0, 0, 0, 0, 0, HWND_MESSAGE, None, None, None)
                if not hwnd:
                    return

                # Register Precision Touchpad (UsagePage 0x0D Digitizer, Usage 0x05 Touch Pad)
                rid = RAWINPUTDEVICE(0x000D, 0x0005, 0x00000100, hwnd)
                if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                    return

                state.abs_trackpad_active = True
                dev_bounds = {}
                msg = MSG()

                while state.running and user32.GetMessageW(ctypes.byref(msg), hwnd, 0, 0) > 0:
                    if msg.message == 0x00FF:  # WM_INPUT
                        dw_size = ctypes.c_uint(0)
                        user32.GetRawInputData(
                            ctypes.c_void_p(msg.lParam), 0x10000003, None, ctypes.byref(dw_size), ctypes.sizeof(RAWINPUTHEADER)
                        )
                        if dw_size.value == 0:
                            continue
                        buf = (ctypes.c_ubyte * dw_size.value)()
                        if user32.GetRawInputData(
                            ctypes.c_void_p(msg.lParam), 0x10000003, buf, ctypes.byref(dw_size), ctypes.sizeof(RAWINPUTHEADER)
                        ) != dw_size.value:
                            continue

                        raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                        if raw.header.dwType != 2:  # RIM_TYPEHID
                            continue

                        h_dev = raw.header.hDevice
                        if h_dev not in dev_bounds:
                            pp_size = ctypes.c_uint(0)
                            user32.GetRawInputDeviceInfoW(h_dev, 0x20000005, None, ctypes.byref(pp_size))
                            if pp_size.value == 0:
                                continue
                            pp_buf = (ctypes.c_ubyte * pp_size.value)()
                            user32.GetRawInputDeviceInfoW(h_dev, 0x20000005, pp_buf, ctypes.byref(pp_size))

                            num_caps = ctypes.c_ushort(32)
                            vcaps = (HIDP_VALUE_CAPS * 32)()
                            hid.HidP_GetValueCaps(0, vcaps, ctypes.byref(num_caps), pp_buf)
                            xmin, xmax, ymin, ymax = 0, 4096, 0, 4096
                            for idx_c in range(num_caps.value):
                                c = vcaps[idx_c]
                                if c.UsagePage == 0x01 and c.UsageMin == 0x30 and c.LogicalMax > c.LogicalMin:
                                    xmin, xmax = c.LogicalMin, c.LogicalMax
                                elif c.UsagePage == 0x01 and c.UsageMin == 0x31 and c.LogicalMax > c.LogicalMin:
                                    ymin, ymax = c.LogicalMin, c.LogicalMax
                            dev_bounds[h_dev] = (pp_buf, xmin, xmax, ymin, ymax)

                        pp_buf, xmin, xmax, ymin, ymax = dev_bounds[h_dev]
                        report_ptr = ctypes.cast(
                            ctypes.addressof(raw) + ctypes.sizeof(RAWINPUTHEADER) + 8, ctypes.POINTER(ctypes.c_ubyte)
                        )
                        rep_len = raw.hid.dwSizeHid

                        val_x = ctypes.c_ulong(0)
                        val_y = ctypes.c_ulong(0)
                        res_x = hid.HidP_GetUsageValue(0, 0x01, 0, 0x30, ctypes.byref(val_x), pp_buf, report_ptr, rep_len)
                        res_y = hid.HidP_GetUsageValue(0, 0x01, 0, 0x31, ctypes.byref(val_y), pp_buf, report_ptr, rep_len)

                        # Check Tip Switch (finger touching touchpad surface)
                        usage_len = ctypes.c_ulong(8)
                        usages = (ctypes.c_ushort * 8)()
                        hid.HidP_GetUsages(0, 0x0D, 0, usages, ctypes.byref(usage_len), pp_buf, report_ptr, rep_len)
                        touching = any(usages[k] == 0x42 for k in range(usage_len.value))

                        if touching and res_x == 0x00110000 and res_y == 0x00110000:
                            u = (val_x.value - xmin) / max(1, (xmax - xmin))
                            v = (val_y.value - ymin) / max(1, (ymax - ymin))
                            teleport_cursor_from_trackpad_uv(u, v, gallery)
                            if not state.menu_open:
                                state.is_pressing = True
                        elif not touching:
                            state.is_pressing = False
            except Exception as e:
                print(f"Windows touchpad raw input fallback to ClipCursor: {e}")

        t = threading.Thread(target=_win_touchpad_thread, daemon=True)
        t.start()


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
            (80, 95, 115),
            1,
        )
        self.centered_canvas = canvas
        self.img_rect = (x_off, y_off, disp_w, disp_h)

        # Physically lock the OS cursor to the new picture rectangle
        clip_os_cursor_to_rect(self.img_rect)
        state.cursor_x = int(np.clip(state.cursor_x, x_off, x_off + disp_w - 1))
        state.cursor_y = int(np.clip(state.cursor_y, y_off, y_off + disp_h - 1))

    def add_image_via_dialog(self):
        release_os_cursor_clip()
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
        clip_os_cursor_to_rect(self.img_rect)


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

    # 1. COLOUR SPECTRUMS -> HARMONIOUS CHORDS + RED/WARM 7TH CHORD ALERT
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_vals = hsv_roi[:, :, 0][mask].astype(np.float32)
    s_vals = hsv_roi[:, :, 1][mask].astype(np.float32) / 255.0
    v_vals = hsv_roi[:, :, 2][mask].astype(np.float32) / 255.0

    chroma_weight = s_vals * np.clip(v_vals * 1.5, 0.05, 1.0)

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

    # 2. PIXEL DENSITY -> MELODIC RHYTHM
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    stipple = cv2.absdiff(gray, blur)[mask]
    stipple_ratio = float(np.mean(stipple > 10))
    bright_ratio = float(np.mean(v_vals > 0.32))
    variance_norm = float(np.clip(np.std(v_vals) * 3.0, 0.0, 1.0))

    density = float(np.clip(bright_ratio * 0.45 + stipple_ratio * 1.7 + variance_norm * 0.35, 0.0, 1.0))
    state.pixel_density = density
    alert_boost = 1.0 + state.warm_alert_7th * 0.35
    state.rhythm_hz = (1.4 + (density ** 1.3) * 5.8) * alert_boost

    # 3. VISUAL SHAPES -> CLEAN SYNTHESIZER PARAMETERS
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
    """Pure, Pitch-Stable Melodic Synthesizer with Warm 7th Alert & Minimal Room Reflection."""
    a = 0.15
    state.s_density += a * (state.pixel_density - state.s_density)
    state.s_alert_7th += a * (state.warm_alert_7th - state.s_alert_7th)
    state.s_elongation += a * (state.shape_elongation - state.s_elongation)
    state.s_jaggedness += a * (state.shape_jaggedness - state.s_jaggedness)
    state.s_angle += a * (state.shape_angle - state.s_angle)

    target = 1.0 if (state.is_pressing and not state.menu_open) else 0.0
    rate = 0.0028 if target > 0.5 else 0.00065
    t_idx = np.arange(1, frames + 1, dtype=np.float32)
    env = state.pad_env + (target - state.pad_env) * (1.0 - (1.0 - rate) ** t_idx)
    state.pad_env = float(env[-1])

    chord_bank = CHORDS_WARM_KEY if state.preset == 0 else CHORDS_DEEP_KEY
    chord_freqs = chord_bank[state.active_chord_idx][1]

    # MELODIC RHYTHM GENERATOR
    rhythm_inc = state.rhythm_hz / SAMPLE_RATE
    r_phases = state.rhythm_phase + np.arange(1, frames + 1, dtype=np.float64) * rhythm_inc
    triggers = np.floor(r_phases) > np.floor(np.insert(r_phases[:-1], 0, state.rhythm_phase))
    state.rhythm_phase = float(r_phases[-1] % 1.0)
    state.visual_pulse = float(np.exp(-state.rhythm_phase * 4.5)) if state.is_pressing else 0.0

    if np.any(triggers) and state.is_pressing and not state.menu_open:
        slot = state.next_note_slot
        if state.s_alert_7th > 0.28:
            melody_pool = [chord_freqs[0], chord_freqs[1], chord_freqs[2], chord_freqs[4], chord_freqs[3], chord_freqs[4] * 2.0]
        else:
            melody_pool = [chord_freqs[0], chord_freqs[1], chord_freqs[2], chord_freqs[3], chord_freqs[1] * 2.0]

        if state.s_angle > 0.6:
            idx = state.melody_step % len(melody_pool)
        elif state.s_angle < 0.35:
            idx = (len(melody_pool) - 1) - (state.melody_step % len(melody_pool))
        else:
            cycle = len(melody_pool) * 2 - 2
            pos = state.melody_step % max(1, cycle)
            idx = pos if pos < len(melody_pool) else cycle - pos

        state.melody_step += 1
        state.note_freqs[slot] = melody_pool[idx]
        state.note_envs[slot] = min(1.0, 0.40 + state.s_density * 0.45 + state.s_alert_7th * 0.15)
        state.note_pans[slot] = float(np.clip(0.5 + 0.32 * np.sin(state.melody_step * 1.5), 0.15, 0.85))
        state.next_note_slot = (slot + 1) % 6

    # WARM HARMONIOUS PAD SYNTHESIS
    pad_L = np.zeros(frames, dtype=np.float64)
    pad_R = np.zeros(frames, dtype=np.float64)

    voice_gains = [
        0.28,
        0.22,
        0.22,
        0.15,
        0.38 * state.s_alert_7th,
    ]

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

        wave = np.sin(p_fund) + octave_harmonic_mix * np.sin(p_oct)

        if i == 4 and state.s_alert_7th > 0.2:
            beacon_pulse = 0.75 + 0.25 * np.sin(2.0 * np.pi * r_phases)
            wave = wave * beacon_pulse

        pan = 0.25 + 0.12 * i
        pad_L += wave * (1.0 - pan) * voice_gains[i]
        pad_R += wave * pan * voice_gains[i]

    pad_L *= env
    pad_R *= env

    # MELODIC PLUCK VOICES
    mel_L = np.zeros(frames, dtype=np.float64)
    mel_R = np.zeros(frames, dtype=np.float64)

    decay_rate = 0.99935 - state.s_elongation * 0.00035 - state.s_density * 0.00025
    decay_curve = decay_rate ** t_idx

    for i in range(6):
        if state.note_envs[i] > 0.001:
            n_env = state.note_envs[i] * decay_curve
            state.note_envs[i] = float(n_env[-1])
            n_inc = (2.0 * np.pi * state.note_freqs[i]) / SAMPLE_RATE
            n_phase = state.note_phases[i] + t_idx * n_inc
            state.note_phases[i] = float(n_phase[-1] % (2.0 * np.pi))

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

    # Minimal Room Reflection (18% feedback, 12% wet mix)
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
    """Renders the full-screen interface with unified PC+Lens cursor locked inside the picture."""
    frame = gallery.centered_canvas.copy()
    ix, iy, iw, ih = gallery.img_rect

    # Top Info Bar (Controlled via Hotkeys so cursor never needs to leave the picture)
    cv2.rectangle(frame, (0, 0), (WIN_W, TOP_BAR_H), (22, 24, 30), -1)
    cv2.line(frame, (0, TOP_BAR_H), (WIN_W, TOP_BAR_H), (60, 65, 75), 1)

    menu_col = (95, 150, 245) if state.menu_open else (45, 52, 65)
    cv2.rectangle(frame, (16, 10), (195, 48), menu_col, -1, cv2.LINE_AA)
    cv2.putText(frame, "Images Menu [M]", (26, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (207, 10), (365, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, "+ Add Pic [O]", (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 235, 210), 1, cv2.LINE_AA)

    p_name = "Key: C Maj Warm [SPACE]" if state.preset == 0 else "Key: F Maj Deep [SPACE]"
    p_col = (60, 100, 75) if state.preset == 0 else (85, 65, 105)
    cv2.rectangle(frame, (377, 10), (620, 48), p_col, -1, cv2.LINE_AA)
    cv2.putText(frame, p_name, (389, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)

    mode_txt = "Absolute Trackpad: ACTIVE (1:1 Corners)" if state.abs_trackpad_active else "Cursor Locked to Picture"
    cv2.rectangle(frame, (632, 10), (985, 48), (45, 75, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, mode_txt, (644, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (190, 245, 215), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (997, 10), (1120, 48), (45, 52, 65), -1, cv2.LINE_AA)
    cv2.putText(frame, f"Zoom: {gallery.zoom:.1f}x [R]", (1007, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 220, 150), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (WIN_W - 120, 10), (WIN_W - 16, 48), (60, 45, 155), -1, cv2.LINE_AA)
    cv2.putText(frame, "Quit [Q]", (WIN_W - 98, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if not state.menu_open:
        # Corner brackets showing the locked picture boundary
        corner_len = 20
        c_col = (120, 235, 180)
        cv2.line(frame, (ix, iy), (ix + corner_len, iy), c_col, 2)
        cv2.line(frame, (ix, iy), (ix, iy + corner_len), c_col, 2)
        cv2.line(frame, (ix + iw - 1, iy), (ix + iw - 1 - corner_len, iy), c_col, 2)
        cv2.line(frame, (ix + iw - 1, iy), (ix + iw - 1, iy + corner_len), c_col, 2)
        cv2.line(frame, (ix, iy + ih - 1), (ix + corner_len, iy + ih - 1), c_col, 2)
        cv2.line(frame, (ix, iy + ih - 1), (ix, iy + ih - 1 - corner_len), c_col, 2)
        cv2.line(frame, (ix + iw - 1, iy + ih - 1), (ix + iw - 1 - corner_len, iy + ih - 1), c_col, 2)
        cv2.line(frame, (ix + iw - 1, iy + ih - 1), (ix + iw - 1, iy + ih - 1 - corner_len), c_col, 2)

        # Telemetry Panel
        panel_x, panel_y = ix + 16, iy + 16
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 465, panel_y + 112), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        border_c = (80, 110, 255) if state.s_alert_7th > 0.35 else (70, 75, 90)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 465, panel_y + 112), border_c, 1)

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

        # Reactive Cursor Lens (Drawn at exact PC cursor location)
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
            f"Map: {title} | Touch Trackpad = Instant 1:1 Jump & Play | [M] Image Menu | [O] Add File | [+/-] Zoom | [Q] Quit",
            (20, WIN_H - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (165, 170, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        # Image Selection Menu Overlay (Positioned inside the picture bounds so locked cursor can click cards)
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


def toggle_fullscreen(win_name, gallery):
    state.fullscreen = not state.fullscreen
    prop = cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
    clip_os_cursor_to_rect(gallery.img_rect)


def make_mouse_handler(gallery):
    def on_mouse(event, raw_x, raw_y, flags, param):
        ix, iy, iw, ih = gallery.img_rect

        # Physically clamp the OS PC cursor inside the picture rectangle so it never leaves the picture
        clamped_x = int(np.clip(raw_x, ix, ix + iw - 1))
        clamped_y = int(np.clip(raw_y, iy, iy + ih - 1))

        if raw_x != clamped_x or raw_y != clamped_y:
            warp_os_cursor(clamped_x, clamped_y)

        # PC cursor and Image Viewer cursor are always identical
        state.cursor_x, state.cursor_y = clamped_x, clamped_y

        if event == cv2.EVENT_MOUSEWHEEL and not state.menu_open:
            if flags > 0:
                gallery.zoom_at(1.22, clamped_x, clamped_y)
            else:
                gallery.zoom_at(1.0 / 1.22, clamped_x, clamped_y)
            analyze_lens(gallery.centered_canvas, clamped_x, clamped_y, state.lens_radius, gallery.img_rect)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if state.menu_open:
                mx1, my1 = (WIN_W - 980) // 2, 95
                cols, card_w, card_h = 3, 280, 210
                start_x, start_y, gap_x, gap_y = mx1 + 35, my1 + 70, 35, 35
                for idx in range(min(6, len(gallery.entries))):
                    r, c = divmod(idx, cols)
                    cx = start_x + c * (card_w + gap_x)
                    cy = start_y + r * (card_h + gap_y)
                    if cx <= clamped_x <= cx + card_w and cy <= clamped_y <= cy + card_h:
                        gallery.select_image(idx)
                        state.menu_open = False
                        return
                return

            state.is_pressing = True
            analyze_lens(gallery.centered_canvas, clamped_x, clamped_y, state.lens_radius, gallery.img_rect)

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.is_pressing and not state.menu_open:
                analyze_lens(gallery.centered_canvas, clamped_x, clamped_y, state.lens_radius, gallery.img_rect)

        elif event == cv2.EVENT_LBUTTONUP:
            if not state.abs_trackpad_active:
                state.is_pressing = False

    return on_mouse


def main():
    gallery = ImageGallery()
    win_name = "Melodic Earth Sonifier"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_name, make_mouse_handler(gallery))

    # Place PC cursor at the center of the picture on launch and lock it to the picture
    ix, iy, iw, ih = gallery.img_rect
    state.cursor_x, state.cursor_y = ix + iw // 2, iy + ih // 2
    warp_os_cursor(state.cursor_x, state.cursor_y)
    clip_os_cursor_to_rect(gallery.img_rect)

    # Start hardware-level absolute trackpad touch listener
    start_absolute_trackpad_listener(gallery)

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype="float32",
        callback=audio_callback,
    )

    try:
        with stream:
            while state.running:
                if state.needs_lens_update and not state.menu_open:
                    state.needs_lens_update = False
                    analyze_lens(
                        gallery.centered_canvas,
                        state.cursor_x,
                        state.cursor_y,
                        state.lens_radius,
                        gallery.img_rect,
                    )

                frame = draw_interface(gallery)
                cv2.imshow(win_name, frame)

                key = cv2.waitKeyEx(16)
                if key == -1:
                    continue

                low_key = key & 0xFF
                if low_key in (ord("q"), ord("Q"), 27):
                    break
                elif low_key in (ord("f"), ord("F")):
                    toggle_fullscreen(win_name, gallery)
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
    finally:
        state.running = False
        release_os_cursor_clip()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()