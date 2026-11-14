import { useEffect, useState } from "react";
import { getJukeboxApi, type Dataset } from "./api";
import { useTrackpad } from "./useTrackpad";

const fallbackDatasets: Dataset[] = [
  ["Aerosols", "A placeholder view of atmospheric aerosol movement.", "#596f4d", "#d9a441"],
  ["Ocean Currents", "A placeholder view of surface ocean circulation.", "#244c63", "#48d1cc"],
  ["Fire Weather", "A placeholder view of regional wildfire risk.", "#693e36", "#e58b52"],
  ["Global CO₂", "A placeholder view of atmospheric carbon dioxide.", "#3e4d72", "#b8c5ed"],
  ["World Topography", "A placeholder tactile map reference.", "#566c4b", "#d0b27a"],
].map(([title, description, background, accent], index) => ({
  id: `fallback-${index}`,
  title,
  description,
  image: `data:image/svg+xml,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450"><rect width="800" height="450" fill="${background}"/><circle cx="620" cy="125" r="72" fill="${accent}" opacity=".5"/><path d="M0 330 Q180 180 360 310 T800 180 V450 H0Z" fill="${accent}" opacity=".72"/></svg>`,
  )}`,
  unit: "value",
  min: 0,
  max: 100,
}));

export default function App() {
  const { pad, onFallbackPointer } = useTrackpad();
  const [datasets, setDatasets] = useState<Dataset[]>(fallbackDatasets);
  const [selectedId, setSelectedId] = useState(fallbackDatasets[0].id);

  useEffect(() => {
    const api = getJukeboxApi();
    if (!api) return;
    void api.get_datasets().then((items) => {
      if (items.length > 0) {
        setDatasets(items);
        setSelectedId(items[0].id);
      }
    });
  }, []);

  const selected = datasets.find((dataset) => dataset.id === selectedId) ?? datasets[0];
  if (!selected) return null;

  return (
    <div className="min-h-screen bg-[#a8bc9c] text-[#263525]">
      <header className="bg-[#4b613f] px-8 py-3">
        <h1 className="font-display text-3xl text-[#e2a95c] sm:text-4xl">TerraGroove</h1>
      </header>
      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-5 px-8 py-6 lg:grid-cols-[minmax(0,1fr)_96px]">
        <section>
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
          </div>
        </section>
        <aside className="flex gap-4 overflow-x-auto lg:flex-col lg:overflow-visible">
          {datasets.map((dataset) => (
            <button
              className={`h-20 min-w-24 overflow-hidden rounded-lg border-4 bg-white ${
                dataset.id === selected.id ? "border-[#4b613f]" : "border-white/70"
              }`}
              key={dataset.id}
              type="button"
              onClick={() => setSelectedId(dataset.id)}
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
