export type Dataset = {
  id: string;
  title: string;
  description: string;
  image: string;
  unit: string;
  min: number;
  max: number;
};

export type JukeboxApi = {
  calibrate_pad: () => Promise<string>;
  get_datasets: () => Promise<Dataset[]>;
  load_dataset: (datasetId: string) => Promise<Dataset>;
  update_pad: (
    x: number,
    y: number,
    touching: boolean,
    fingers: number,
  ) => Promise<void>;
};

declare global {
  interface Window {
    pywebview?: {
      api: JukeboxApi;
    };
  }
}

export function getJukeboxApi(): JukeboxApi | undefined {
  return window.pywebview?.api;
}
