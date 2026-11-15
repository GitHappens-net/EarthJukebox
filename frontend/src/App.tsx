import { useEffect, useState } from "react";
import { CircleHelp, Compass, Info } from "lucide-react";
import { getJukeboxApi, type Dataset } from "./api";
import AboutPage from "./pages/AboutPage";
import ExplorePage from "./pages/ExplorePage";
import InstructionsPage from "./pages/InstructionsPage";
import { useTrackpad } from "./useTrackpad";

type Page = "explore" | "about" | "instructions";

export default function App() {
  const { pad, onFallbackPointer } = useTrackpad();
  const [page, setPage] = useState<Page>("explore");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [error, setError] = useState<string>();
  const [audioEnabled, setAudioEnabled] = useState(true);

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
        setSelectedId((current) => (
          current
          ?? items.find((item) => item.title.toLowerCase().includes("geos"))?.id
          ?? items[0].id
        ));
      }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
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
      if (event.key === " ") {
        event.preventDefault();
        setAudioEnabled((enabled) => !enabled);
        return;
      }
      if (event.key.toLowerCase() === "c") {
        event.preventDefault();
        void getJukeboxApi()?.calibrate_pad();
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
      void getJukeboxApi()?.set_audio_enabled(page === "explore" && audioEnabled);
    };
    updateAudioState();
    window.addEventListener("pywebviewready", updateAudioState);
    return () => window.removeEventListener("pywebviewready", updateAudioState);
  }, [audioEnabled, page]);

  const selected = datasets.find((dataset) => dataset.id === selectedId);
  const selectDataset = (dataset: Dataset) => {
    setSelectedId(dataset.id);
    void getJukeboxApi()?.select_dataset(dataset.id);
  };

  return (
    <div className="bg-[#FFF2B4] min-h-screen">
      <header className="relative z-10 bg-[#15370D]">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
          <button className="text-left" type="button" onClick={() => setPage("explore")}>
            <span className="font-display text-3xl text-white">TERRA GROOVE</span>
          </button>
          <nav className="flex gap-1 rounded-full bg-black/10 p-1 text-sm text-white" aria-label="Primary navigation">
            {([
              ["explore", "Explore", Compass],
              ["about", "About", Info],
              ["instructions", "Instructions", CircleHelp],
            ] as const).map(([id, label, Icon]) => (
              <button
                className={`rounded-full px-3 py-1.5 font-bold transition sm:px-4 ${page === id ? "bg-[#E5CD82] text-[#15370D] shadow-sm" : "hover:bg-white/10 hover:text-white"}`}
                key={id}
                type="button"
                onClick={() => setPage(id)}
              >
                <Icon className="mr-1.5 mb-[2px] inline-block align-text-bottom" size={16} aria-hidden="true" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main
        id="main-content"
        className={`relative z-10 mx-auto px-4 py-5 sm:px-6 lg:px-12 ${
          page === "explore" ? "max-w-[1600px]" : "max-w-7xl"
        }`}
        tabIndex={-1}
      >
        {page === "explore" && (
          <ExplorePage
            datasets={datasets}
            selected={selected}
            selectedId={selectedId}
            audioEnabled={audioEnabled}
            pad={pad}
            error={error}
            onSelect={selectDataset}
            onToggleAudio={() => setAudioEnabled((enabled) => !enabled)}
            onFallbackPointer={onFallbackPointer}
          />
        )}
        {page === "about" && <AboutPage />}
        {page === "instructions" && <InstructionsPage />}
      </main>
    </div>
  );
}
