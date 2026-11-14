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
python3 backend/main.py --input evdev --device /dev/input/event8
python3 backend/main.py --input browser
```

Browser mode is the portable pointer-level fallback. Raw multi-touch hardware
events remain platform-specific and require a native Windows Pointer API
adapter if individual fingers or gestures are needed.
