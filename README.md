# TerraGroove

## Cross-platform touchpad input

Run the app with automatic input selection:

```bash
python3 backend/main.py --input auto
```

On Linux, `auto` uses raw `evdev` events. On Windows and other platforms, it
uses browser pointer events from the pywebview frontend, so a touchpad works
through its normal mouse/pointer interface without Linux-specific dependencies.
You can select either mode explicitly:

```bash
python3 backend/main.py --input evdev
python3 backend/main.py --input browser
```

Browser mode is the portable pointer-level fallback. Raw multi-touch hardware
events remain platform-specific and require a native Windows Pointer API
adapter if individual fingers or gestures are needed.

On Fedora/Linux, pywebview can use the installed PySide6 WebEngine backend:

the launcher prefers PySide6 when it is available, avoiding conflicts between
system Qt and separately installed PyQt packages. If PySide6 WebEngine is not
installed, install the matching PySide6 WebEngine package for your Python
environment.

The music, vision, dataset, and audio-rendering backend lives in
`backend/earth_jukebox/`; `backend/main.py` exposes it to the React frontend
through pywebview.
