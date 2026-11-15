import type { PointerEvent } from "react";
import {
  Database,
  RotateCcw,
  Volume2,
  VolumeX,
} from "lucide-react";
import { getJukeboxApi, type Dataset } from "../api";

function Legend({ dataset }: { dataset: Dataset }) {
  if (dataset.legend.length === 0) {
    return <p className="text-sm text-black/40">This reference image has no color scale.</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-2 text-sm text-black/50 sm:grid-cols-3">
      {dataset.legend.map((item) => (
        <div
          className="flex min-w-0 items-center gap-2 rounded-lg bg-white/35 px-2 py-1.5"
          key={`${item.color}-${item.meaning}`}
          title={`${item.color}: ${item.meaning}`}
          aria-label={`${item.color}: ${item.meaning}`}
        >
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-full ring-1 ring-black/10"
            style={{ backgroundColor: item.color.toLowerCase() }}
            aria-hidden="true"
          />
          <strong className="truncate font-semibold text-black/65">{item.color}</strong>
        </div>
      ))}
    </div>
  );
}

export type ExplorePageProps = {
  datasets: Dataset[];
  selected?: Dataset;
  selectedId?: string;
  audioEnabled: boolean;
  pad: { x: number; y: number; touching: boolean };
  error?: string;
  onSelect: (dataset: Dataset) => void;
  onToggleAudio: () => void;
  onFallbackPointer: (event: PointerEvent<HTMLElement>, touching: boolean) => void;
};

export default function ExplorePage({
  datasets,
  selected,
  selectedId,
  audioEnabled,
  pad,
  error,
  onSelect,
  onToggleAudio,
  onFallbackPointer,
}: ExplorePageProps) {
  if (!selected) {
    return (
      <div className="bg-[#E5CD82]/60 rounded-2xl shadow-md flex min-h-72 items-center justify-center p-8 text-black/40 text-xl font-semibold">
        {error ?? "Loading NASA datasets from the backend…"}
      </div>
    );
  }

  return (
    <div className="grid gap-4 lg:items-stretch lg:grid-cols-[minmax(0,1fr)_minmax(280px,320px)]">
      <section className="min-w-0">
        <div
          className={`trackpad-shell relative aspect-video w-full touch-none overflow-hidden rounded-3xl ${
            pad.touching ? "trackpad-active" : ""
          } ${audioEnabled ? "cursor-none" : ""}`}
          onPointerDown={(event) => onFallbackPointer(event, true)}
          onPointerMove={(event) => {
            if (event.buttons > 0) onFallbackPointer(event, true);
          }}
          onPointerUp={(event) => onFallbackPointer(event, false)}
          onPointerLeave={(event) => onFallbackPointer(event, false)}
          aria-label="Interactive NASA dataset trackpad"
        >
          <img className="h-full w-full object-cover" src={selected.image} alt={selected.title} />
          <div className="absolute inset-0 bg-black/10" />
          <div
            className={`cursor-dot absolute z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white ${
              audioEnabled ? "" : "hidden"
            }`}
            style={{ left: `${pad.x * 100}%`, top: `${pad.y * 100}%` }}
            aria-hidden="true"
          />
          <button
            className="glass-button control-button absolute bottom-4 right-4 z-20 inline-flex h-10 w-10 items-center justify-center p-0 text-white"
            type="button"
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => void getJukeboxApi()?.calibrate_pad()}
            aria-label="Recenter trackpad cursor"
            title="Recenter trackpad cursor"
          >
            <RotateCcw size={16} aria-hidden="true" />
          </button>
          <button
            className="glass-button control-button absolute bottom-4 right-16 z-20 inline-flex h-10 w-10 items-center justify-center p-0 text-white"
            type="button"
            onPointerDown={(event) => event.stopPropagation()}
            onClick={onToggleAudio}
            aria-pressed={audioEnabled}
            aria-label={audioEnabled ? "Mute audio" : "Unmute audio"}
            title={audioEnabled ? "Mute audio" : "Unmute audio"}
          >
            {audioEnabled ? <Volume2 size={16} aria-hidden="true" /> : <VolumeX size={16} aria-hidden="true" />}
          </button>
        </div>

        <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(240px,0.8fr)]">
          <article className="rounded-2xl bg-[#E5CD82]/60 p-4 shadow-md">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="eyebrow">About this dataset</p>
                <h2 className="mt-1 font-display text-xl text-white text-shadow-md">{selected.title}</h2>
              </div>
            </div>
            <p className="mt-3 text-lg leading-relaxed text-black/40">{selected.description}</p>
            {error && <p className="mt-4 text-sm text-rose-300" role="alert">{error}</p>}
          </article>
          <article className="rounded-2xl bg-[#E5CD82]/60 p-4 shadow-md">
            <p className="eyebrow">Legend</p>
            <h2 className="mt-1 font-display text-xl text-white text-shadow-md">Reading the image</h2>
            <div className="mt-4">
              <Legend dataset={selected} />
            </div>
          </article>
        </div>
      </section>

      <aside className="flex min-w-0 flex-col rounded-2xl bg-[#E5CD82]/60 p-2 shadow-md lg:min-h-full" aria-label="Dataset selection">
        <div className="mb-2 px-2">
          <p className="eyebrow">The collection</p>
          <h2 className="flex items-center gap-2 font-display text-xl text-white text-shadow-md">
            <Database size={19} strokeWidth={1.8} aria-hidden="true" />
            Earth datasets
          </h2>
        </div>
        <div className="scrollbar-hidden flex gap-2 overflow-x-auto lg:min-h-0 lg:flex-1 lg:block lg:space-y-2 lg:overflow-y-auto">
          {datasets.map((dataset) => (
            <button
              className={`dataset-card flex min-w-56 items-center gap-3 rounded-xl p-2 text-left lg:w-full ${
                dataset.id === selectedId ? "dataset-card-selected" : ""
              }`}
              key={dataset.id}
              type="button"
              onClick={() => onSelect(dataset)}
              aria-label={`Select dataset ${dataset.title}`}
              aria-pressed={dataset.id === selectedId}
              aria-keyshortcuts={dataset.id}
            >
              <img className="h-14 w-20 shrink-0 rounded-lg object-cover" src={dataset.image} alt="" />
              <span className="min-w-0">
                <strong className="block truncate text-sm text-white">{dataset.title}</strong>
                <span className="mt-1 block truncate text-xs text-emerald-300/70">{dataset.category}</span>
                <span className="mt-1 block truncate text-xs text-slate-400">{dataset.description}</span>
              </span>
            </button>
          ))}
        </div>
      </aside>
    </div>
  );
}
