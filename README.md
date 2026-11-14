# EarthJukebox

## Trackpad UI

The trackpad monitor uses a React/TypeScript frontend hosted by pywebview.

Install the Python dependencies:

```bash
python3 -m pip install pywebview evdev
```

Build the UI:

```bash
cd TESTS
npm install
npm run build
```

Run the desktop app from the repository root:

```bash
python3 TESTS/main.py --device /dev/input/event7
```

Omit `--device` to auto-detect a single touchpad. For frontend development,
run `npm run dev` in `TESTS` and start the Python app with `--dev`.
