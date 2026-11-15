export type Dataset = {
  id: string;
  title: string;
  description: string;
  category: string;
  image: string;
  unit: string;
  min: number;
  max: number;
  legend: { color: string; meaning: string }[];
};

export type JukeboxApi = {
  calibrate_pad: () => Promise<string>;
  get_datasets: () => Promise<Dataset[]>;
  load_dataset: (datasetId: string) => Promise<Dataset>;
  select_dataset: (datasetId: string) => Promise<void>;
  update_pad: (
    x: number,
    y: number,
    touching: boolean,
    fingers: number,
  ) => Promise<void>;
  set_audio_enabled: (enabled: boolean) => Promise<void>;
  get_music_snapshot: () => Promise<MusicSnapshot>;
  get_audio_status: () => Promise<AudioStatus>;
};

export type MusicSnapshot = {
  analysis?: {
    dominant_color?: string;
    intensity?: number;
    density?: number;
    shape?: string;
  };
  music?: { chord?: string; bpm?: number; primary_voice?: { note_name?: string } };
  description?: string;
};

export type AudioStatus = {
  backend: string;
  running: boolean;
  bpm: number;
};

declare global {
  interface Window {
    pywebview?: {
      api: JukeboxApi;
    };
  }
}

export function getJukeboxApi(): JukeboxApi | undefined {
  const api = window.pywebview?.api;
  if (!api || typeof api.get_datasets !== "function") {
    return undefined;
  }
  return api;
}
