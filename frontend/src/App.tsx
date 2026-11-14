import { useState } from "react";
import { getJukeboxApi } from "./api";
import { useTrackpad } from "./useTrackpad";

export default function App() {
  const { pad, isDesktopApp, onFallbackPointer } = useTrackpad();
  const [message, setMessage] = useState("Move your finger across the trackpad");

  const calibrate = async () => {
    const api = getJukeboxApi();
    if (!api) {
      setMessage("Calibration is available in the desktop app");
      return;
    }
    setMessage(await api.calibrate_pad());
  };

  return (
    <main className="mx-auto max-w-4xl px-6 py-10 text-[#e8f2f1]">
      <header>
        <p className="font-space text-xs uppercase tracking-[0.18em] text-[#48d1cc]">
          Earth Information Jukebox
        </p>
      </header>
      <section
        className="relative my-10 min-h-[260px] h-[min(52vw,440px)] touch-none overflow-hidden rounded-2xl border border-[#2e5054] bg-[radial-gradient(circle_at_center,_#193337,_#152124_65%)]"
        onPointerDown={(event) => onFallbackPointer(event, true)}
        onPointerMove={(event) => {
          if (event.buttons > 0) onFallbackPointer(event, true);
        }}
        onPointerUp={(event) => onFallbackPointer(event, false)}
        onPointerLeave={(event) => onFallbackPointer(event, false)}
      >
        <div className="absolute inset-[12%] rounded-full border border-[#286267] opacity-50" />
        <div
          className={`absolute z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#48d1cc] transition-[opacity,box-shadow] ${
            pad.touching
              ? "opacity-100 shadow-[0_0_0_12px_#48d1cc33,0_0_35px_#48d1cc]"
              : "opacity-30 shadow-[0_0_0_8px_#48d1cc22,0_0_28px_#48d1cc]"
          }`}
          style={{ left: `${pad.x * 100}%`, top: `${pad.y * 100}%` }}
        />
      </section>
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-xl border border-[#294146] bg-[#152326] p-4">
          <span className="block text-xs uppercase text-[#91a5a6]">X</span>
          <strong className="mt-2 block">{pad.x.toFixed(3)}</strong>
        </div>
        <div className="rounded-xl border border-[#294146] bg-[#152326] p-4">
          <span className="block text-xs uppercase text-[#91a5a6]">Y</span>
          <strong className="mt-2 block">{pad.y.toFixed(3)}</strong>
        </div>
        <div className="rounded-xl border border-[#294146] bg-[#152326] p-4">
          <span className="block text-xs uppercase text-[#91a5a6]">Status</span>
          <strong className="mt-2 block">{pad.touching ? "Touching" : "Idle"}</strong>
        </div>
        <div className="rounded-xl border border-[#294146] bg-[#152326] p-4">
          <span className="block text-xs uppercase text-[#91a5a6]">Fingers</span>
          <strong className="mt-2 block">{pad.fingers}</strong>
        </div>
      </section>
      <button
        className="mt-7 cursor-pointer rounded-lg bg-[#48d1cc] px-5 py-3 font-semibold text-[#10191c]"
        type="button"
        onClick={calibrate}
      >
        Calibrate pad
      </button>
    </main>
  );
}
