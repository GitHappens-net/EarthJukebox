export type Dataset = {
  id: string;
  unit: string;
  min: number;
  max: number;
};

export type JukeboxApi = {
  calibrate_pad: () => Promise<string>;
  load_dataset: (datasetId: string) => Promise<Dataset>;
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
