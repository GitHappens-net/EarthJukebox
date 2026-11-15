import { useEffect, useState } from "react";
import { getJukeboxApi, type Dataset, type MusicSnapshot } from "./api";
import { useTrackpad } from "./useTrackpad";

export default function App() {
  const { pad, onFallbackPointer } = useTrackpad();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [snapshot, setSnapshot] = useState<MusicSnapshot | null>(null);
  const [error, setError] = useState<string>();
  const [audioBackend, setAudioBackend] = useState<string>();

  useEffect(() => {
    const loadDatasets = () => {
      const api = getJukeboxApi();
      if (!api) return;
      void api
        .get_datasets()
        .then((items) => {
          if (items.length === 0) {
            setError("The backend returned no datasets.");
            return;
          }
          setDatasets(items);
          setSelectedId(items[0].id);
        })
        .catch((reason: unknown) => {
          setError(reason instanceof Error ? reason.message : String(reason));
        });

      void api
        .get_audio_status()
        .then((status) => setAudioBackend(status.backend))
        .catch((reason: unknown) => {
          setError(reason instanceof Error ? reason.message : String(reason));
        });
    };

    loadDatasets();
    window.addEventListener("pywebviewready", loadDatasets);
    return () => window.removeEventListener("pywebviewready", loadDatasets);
  }, []);

  useEffect(() => {
    const updateSnapshot = () => {
      const api = getJukeboxApi();
      if (!api) return;
      void api.get_music_snapshot().then(setSnapshot).catch(() => undefined);
    };

    updateSnapshot();
    const interval = window.setInterval(updateSnapshot, 500);
    return () => window.clearInterval(interval);
  }, []);

  const selected = datasets.find((dataset) => dataset.id === selectedId);

  return (
    <div className="min-h-screen bg-[#a8bc9c] text-[#263525]">
      <header className="bg-[#4b613f] px-8 py-3">
        <h1 className="font-display text-3xl text-[#e2a95c] sm:text-4xl">TerraGroove</h1>
      </header>
      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-5 px-8 py-6 lg:grid-cols-[minmax(0,1fr)_96px]">
        <section>
          {!selected && (
            <p className="rounded-lg bg-white/70 p-4" aria-live="polite">
              {error ?? "Loading NASA datasets from the backend…"}
            </p>
          )}
          {selected && (
            <>
              <div
                className="relative aspect-video max-h-[500px] touch-none overflow-hidden rounded-xl border-2 border-white/60 bg-[#d6d6d6] shadow-lg"
                onPointerDown={(event) => onFallbackPointer(event, true)}
                onPointerMove={(event) => {
                  if (event.buttons > 0) onFallbackPointer(event, true);
                }}
                onPointerUp={(event) => onFallbackPointer(event, false)}
                onPointerLeave={(event) => onFallbackPointer(event, false)}
              >
                <img className="h-full w-full object-cover" src={selected.image} alt={selected.title} />
                <div className="absolute inset-0 bg-black/10" />
                <div
                  className="absolute z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white"
                  style={{ left: `${pad.x * 100}%`, top: `${pad.y * 100}%` }}
                />
                <button
                  className="absolute bottom-3 right-3 z-20 rounded-lg bg-[#4b613f] px-3 py-1.5 text-xs font-semibold text-white"
                  type="button"
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={() => void getJukeboxApi()?.calibrate_pad()}
                >
                  Calibrate
                </button>
              </div>
              <div className="mt-5">
                <h2 className="font-display text-2xl text-[#4b613f]">{selected.title}</h2>
                <p className="mt-1 max-w-2xl">{selected.description}</p>
                <p className="mt-2 text-xs text-[#52644c]">
                  Range: {selected.min}–{selected.max} {selected.unit}
                </p>
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <span className="text-sm">
                    {pad.touching ? "Live sonification active" : "Touch the image to play"}
                  </span>
                  {audioBackend && <span className="text-xs">Audio: {audioBackend}</span>}
                  {snapshot?.analysis && (
                    <span className="text-sm" aria-live="polite">
                      {snapshot.analysis.dominant_color} · {snapshot.music?.chord ?? "—"} ·{" "}
                      {snapshot.music?.primary_voice?.note_name ?? "—"}
                    </span>
                  )}
                </div>
                {snapshot?.description && <p className="mt-2 text-xs">{snapshot.description}</p>}
                {error && <p className="mt-2 text-xs text-red-800">{error}</p>}
              </div>
            </>
          )}
        </section>
        <aside className="flex gap-4 overflow-x-auto lg:flex-col lg:overflow-visible">
          {datasets.map((dataset) => (
            <button
              className={`h-20 min-w-24 overflow-hidden rounded-lg border-4 bg-white ${
                dataset.id === selectedId ? "border-[#4b613f]" : "border-white/70"
              }`}
              key={dataset.id}
              type="button"
              onClick={() => {
                setSelectedId(dataset.id);
                void getJukeboxApi()?.select_dataset(dataset.id);
              }}
              title={dataset.title}
            >
              <img className="h-full w-full object-cover" src={dataset.image} alt="" />
            </button>
          ))}
        </aside>
      </main>
    </div>
  );
}
