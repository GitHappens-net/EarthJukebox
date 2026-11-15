import { useEffect, useState, type PointerEvent } from "react";
import { getJukeboxApi, type Dataset, type MusicSnapshot } from "./api";
import { useTrackpad } from "./useTrackpad";

type Page = "explore" | "about" | "instructions";

const swatches: Record<string, string> = {
  red: "#ef4444",
  orange: "#f59e0b",
  yellow: "#facc15",
  green: "#4ade80",
  blue: "#38bdf8",
  violet: "#a78bfa",
  purple: "#a78bfa",
  white: "#f8fafc",
  black: "#17202a",
};

function Legend({ dataset }: { dataset: Dataset }) {
  if (dataset.legend.length === 0) {
    return <p className="text-sm text-slate-400">This reference image has no color scale.</p>;
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {dataset.legend.map((item) => (
        <div className="flex gap-3 text-sm text-slate-300" key={`${item.color}-${item.meaning}`}>
          <span
            className="mt-1 h-3 w-3 shrink-0 rounded-full ring-2 ring-white/20"
            style={{ backgroundColor: swatches[item.color.toLowerCase()] ?? "#94a3b8" }}
          />
          <span>
            <strong className="font-semibold text-white">{item.color}</strong>{" "}
            {item.meaning}
          </span>
        </div>
      ))}
    </div>
  );
}

function ExplorePage({
  datasets,
  selected,
  selectedId,
  snapshot,
  audioBackend,
  pad,
  error,
  onSelect,
  onFallbackPointer,
}: {
  datasets: Dataset[];
  selected?: Dataset;
  selectedId?: string;
  snapshot: MusicSnapshot | null;
  audioBackend?: string;
  pad: { x: number; y: number; touching: boolean };
  error?: string;
  onSelect: (dataset: Dataset) => void;
  onFallbackPointer: (event: PointerEvent<HTMLElement>, touching: boolean) => void;
}) {
  if (!selected) {
    return (
      <div className="glass-panel flex min-h-72 items-center justify-center p-8 text-slate-300">
        {error ?? "Loading NASA datasets from the backend…"}
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_270px]">
      <section className="min-w-0">
        <div
          className={`trackpad-shell relative aspect-video max-h-[560px] touch-none overflow-hidden rounded-3xl ${
            pad.touching ? "trackpad-active" : ""
          }`}
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
          <div className="absolute left-5 top-5 flex items-center gap-2 rounded-full bg-slate-950/50 px-3 py-1.5 text-xs text-white backdrop-blur-md">
            <span className={`h-2 w-2 rounded-full ${pad.touching ? "bg-emerald-400 animate-pulse" : "bg-slate-400"}`} />
            {pad.touching ? "Listening live" : "Touch to explore"}
          </div>
          <div
            className="cursor-dot absolute z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white"
            style={{ left: `${pad.x * 100}%`, top: `${pad.y * 100}%` }}
            aria-hidden="true"
          />
          <button
            className="glass-button absolute bottom-4 right-4 z-20 px-4 py-2 text-xs font-semibold text-white"
            type="button"
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => void getJukeboxApi()?.calibrate_pad()}
            aria-label="Recenter trackpad cursor"
          >
            Recenter pad
          </button>
          <div className="absolute bottom-5 left-5 right-32">
            <p className="eyebrow text-white/70">{selected.category}</p>
            <h1 className="font-display text-3xl text-white sm:text-5xl">{selected.title}</h1>
          </div>
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(280px,0.7fr)]">
          <article className="glass-panel p-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="eyebrow">About this dataset</p>
                <h2 className="mt-1 font-display text-2xl text-white">{selected.title}</h2>
              </div>
              {audioBackend && <span className="status-pill">♫ {audioBackend}</span>}
            </div>
            <p className="mt-4 leading-relaxed text-slate-300">{selected.description}</p>
            <div className="mt-5 flex flex-wrap gap-2 text-xs text-slate-300">
              <span className="info-chip">Unit: {selected.unit}</span>
              {snapshot?.music?.chord && <span className="info-chip">Chord: {snapshot.music.chord}</span>}
              {snapshot?.music?.primary_voice?.note_name && (
                <span className="info-chip">Note: {snapshot.music.primary_voice.note_name}</span>
              )}
            </div>
            {error && <p className="mt-4 text-sm text-rose-300" role="alert">{error}</p>}
            <p className="sr-only" aria-live="polite">
              {pad.touching ? "Live sonification active." : "Sonification paused."}{" "}
              {snapshot?.music?.chord ? `Current chord ${snapshot.music.chord}.` : ""}
            </p>
          </article>
          <article className="glass-panel p-6">
            <p className="eyebrow">Legend</p>
            <h2 className="mt-1 font-display text-2xl text-white">Reading the image</h2>
            <div className="mt-5">
              <Legend dataset={selected} />
            </div>
          </article>
        </div>
      </section>

      <aside className="glass-panel h-fit p-3 lg:sticky lg:top-6" aria-label="Dataset selection">
        <div className="mb-3 px-3">
          <p className="eyebrow">The collection</p>
          <h2 className="font-display text-2xl text-white">Earth datasets</h2>
        </div>
        <div className="scrollbar-hidden flex gap-2 overflow-x-auto lg:block lg:max-h-[calc(100vh-12rem)] lg:space-y-2 lg:overflow-y-auto">
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

function AboutPage() {
  return (
    <div className="glass-panel mx-auto max-w-3xl p-8 sm:p-12">
      <p className="eyebrow">About TerraGroove</p>
      <h1 className="mt-2 font-display text-5xl text-white">Listen to the planet.</h1>
      <p className="mt-6 text-lg leading-relaxed text-slate-300">
        TerraGroove turns NASA Earth observation imagery into a tactile, musical
        landscape. Explore a dataset with the trackpad and the live Python
        sequencer responds to color, texture, intensity, and movement.
      </p>
      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        {["NASA imagery", "Computer vision", "Live MIDI sound"].map((item) => (
          <div className="info-card" key={item}>{item}</div>
        ))}
      </div>
    </div>
  );
}

function InstructionsPage() {
  return (
    <div className="glass-panel mx-auto max-w-3xl p-8 sm:p-12">
      <p className="eyebrow">Quick start</p>
      <h1 className="mt-2 font-display text-5xl text-white">How to explore</h1>
      <ol className="mt-8 space-y-6">
        {[
          ["Choose a dataset", "Select an image from the collection on the left."],
          ["Touch the image", "Move across the large image area. Your position controls the sampled region."],
          ["Listen", "The backend sequencer sonifies the sampled colors and visual features in real time."],
          ["Read the legend", "Use the legend below the image to understand what the colors represent."],
        ].map(([title, text], index) => (
          <li className="flex gap-4" key={title}>
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-400/20 text-sm font-bold text-emerald-300">
              {index + 1}
            </span>
            <span>
              <strong className="text-white">{title}</strong>
              <span className="mt-1 block text-slate-400">{text}</span>
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function App() {
  const { pad, onFallbackPointer } = useTrackpad();
  const [page, setPage] = useState<Page>("explore");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [snapshot, setSnapshot] = useState<MusicSnapshot | null>(null);
  const [error, setError] = useState<string>();
  const [audioBackend, setAudioBackend] = useState<string>();

  useEffect(() => {
    const loadDatasets = () => {
      const api = getJukeboxApi();
      if (!api) return;
      void api.get_datasets().then((items) => {
        if (items.length === 0) {
          setError("The backend returned no datasets.");
          return;
        }
        setDatasets(items);
        setSelectedId((current) => current ?? items[0].id);
      }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
      void api.get_audio_status().then((status) => setAudioBackend(status.backend));
    };
    loadDatasets();
    window.addEventListener("pywebviewready", loadDatasets);
    return () => window.removeEventListener("pywebviewready", loadDatasets);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
        setPage("instructions");
        return;
      }
      if (event.key === "Escape") {
        setPage("explore");
        return;
      }

      const currentIndex = datasets.findIndex((dataset) => dataset.id === selectedId);
      let nextIndex = -1;
      if (/^[1-9]$/.test(event.key)) {
        nextIndex = Number(event.key) - 1;
      } else if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % datasets.length;
      } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        nextIndex = currentIndex <= 0 ? datasets.length - 1 : currentIndex - 1;
      }

      if (nextIndex >= 0 && nextIndex < datasets.length) {
        event.preventDefault();
        const dataset = datasets[nextIndex];
        setPage("explore");
        setSelectedId(dataset.id);
        void getJukeboxApi()?.select_dataset(dataset.id);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [datasets, selectedId]);

  useEffect(() => {
    const updateAudioState = () => {
      void getJukeboxApi()?.set_audio_enabled(page === "explore");
    };
    updateAudioState();
    window.addEventListener("pywebviewready", updateAudioState);
    return () => window.removeEventListener("pywebviewready", updateAudioState);
  }, [page]);

  useEffect(() => {
    const updateSnapshot = () => {
      const api = getJukeboxApi();
      if (api) void api.get_music_snapshot().then(setSnapshot).catch(() => undefined);
    };
    const interval = window.setInterval(updateSnapshot, 500);
    return () => window.clearInterval(interval);
  }, []);

  const selected = datasets.find((dataset) => dataset.id === selectedId);

  return (
    <div className="app-shell min-h-screen">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <div className="ambient-orb ambient-orb-one" />
      <div className="ambient-orb ambient-orb-two" />
      <header className="relative z-10 bg-[#4b613f]">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <button className="text-left" type="button" onClick={() => setPage("explore")}>
            <p className="eyebrow text-[#d5dfcf]">NASA Earth Information</p>
            <span className="font-display text-3xl text-[#e2a95c]">TerraGroove</span>
          </button>
          <nav className="flex gap-1 rounded-lg bg-black/10 p-1 text-sm text-white/80" aria-label="Primary navigation">
            {([["explore", "Explore"], ["about", "About"], ["instructions", "Instructions"]] as const).map(([id, label]) => (
              <button
                className={`rounded-md px-3 py-2 transition sm:px-4 ${page === id ? "bg-[#e2a95c] text-[#263525] shadow-sm" : "hover:bg-white/10 hover:text-white"}`}
                key={id}
                type="button"
                onClick={() => setPage(id)}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main id="main-content" className="relative z-10 mx-auto max-w-7xl px-5 py-8 sm:px-8 sm:py-12" tabIndex={-1}>
        {page === "explore" && (
          <ExplorePage
            datasets={datasets}
            selected={selected}
            selectedId={selectedId}
            snapshot={snapshot}
            audioBackend={audioBackend}
            pad={pad}
            error={error}
            onSelect={(dataset) => {
              setSelectedId(dataset.id);
              void getJukeboxApi()?.select_dataset(dataset.id);
            }}
            onFallbackPointer={onFallbackPointer}
          />
        )}
        {page === "about" && <AboutPage />}
        {page === "instructions" && <InstructionsPage />}
      </main>
    </div>
  );
}
