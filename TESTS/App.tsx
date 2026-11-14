import { useState } from "react";
import { useTrackpad } from "./useTrackpad";

declare global {
  interface Window {
    pywebview?: {
      api: {
        calibrate_pad: () => Promise<string>;
      };
    };
  }
}

export default function App() {
  const { pad, isDesktopApp, onFallbackPointer } = useTrackpad();
  const [message, setMessage] = useState("Move your finger across the trackpad");

  const calibrate = async () => {
    if (!window.pywebview) {
      setMessage("Calibration is available in the desktop app");
      return;
    }
    setMessage(await window.pywebview.api.calibrate_pad());
  };

  return (
    <main>
      <header>
        <p className="eyebrow">Earth Information Jukebox</p>
        <h1>Trackpad monitor</h1>
        <p className="subtitle">
          {isDesktopApp ? "Connected to the Linux touchpad" : "Browser preview mode"}
        </p>
      </header>
      <section
        className="trackpad"
        onPointerDown={(event) => onFallbackPointer(event, true)}
        onPointerMove={(event) => {
          if (event.buttons > 0) onFallbackPointer(event, true);
        }}
        onPointerUp={(event) => onFallbackPointer(event, false)}
        onPointerLeave={(event) => onFallbackPointer(event, false)}
      >
        <div
          className={`cursor ${pad.touching ? "active" : ""}`}
          style={{ left: `${pad.x * 100}%`, top: `${pad.y * 100}%` }}
        />
      </section>
      <section className="readout">
        <div><span>X</span><strong>{pad.x.toFixed(3)}</strong></div>
        <div><span>Y</span><strong>{pad.y.toFixed(3)}</strong></div>
        <div><span>Status</span><strong>{pad.touching ? "Touching" : "Idle"}</strong></div>
        <div><span>Fingers</span><strong>{pad.fingers}</strong></div>
      </section>
      <button type="button" onClick={calibrate}>Calibrate pad</button>
      <p className="message">{message}</p>
    </main>
  );
}
