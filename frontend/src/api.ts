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
  select_dataset: (datasetId: string) => Promise<void>;
  update_pad: (
    x: number,
    y: number,
    touching: boolean,
    fingers: number,
  ) => Promise<void>;
  set_audio_enabled: (enabled: boolean) => Promise<void>;
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
