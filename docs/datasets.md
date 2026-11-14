# NASA EIC Datasets & Tactile Hardware

Earth Jukebox ships with **7 NASA Earth Information Center (EIC) visualizations** in `data/images/` and **1 tactile reference world map + 3D-printable models** in `hardware/tactile-map/`.

`DatasetManager` (configured via `src/earth_jukebox/datasets/catalog.json`) automatically discovers these files, normalizes oversized frames down to a working resolution of at most `960x540` using `cv2.INTER_AREA` while preserving aspect ratio, caches decoded BGR arrays in memory, and binds curated scientific summaries and color-legend meanings for spoken TTS readout (`[R]` and `[I]`).

---

## NASA EIC Visualizations (`data/images/`)

| # | Filename (`data/images/`) | Native Size | Dataset Title & Source | Unit |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `aerosol-optical-depth-pm25-2019.png` | `1007x557` | Global PM2.5 Aerosol Optical Depth (2019) -- MODIS / MISR / SeaWiFS | `ug/m3` |
| 2 | `geos-aerosols-2025-08-19.png` | `1920x1080` | GEOS Global Aerosol Transport (Aug 2025) -- NASA GEOS | `Aerosol Plumes` |
| 3 | `ecco2-ocean-currents-grid.jpg` | `6840x3420` | ECCO2 Perpetual Ocean Currents & Temperature -- NASA ECCO2 | `deg C / Flow` |
| 4 | `fwi-conus-2026-08-01.png` | `1920x1080` | North America Fire Weather Index (Aug 2026) -- NASA GFWED | `FWI Risk` |
| 5 | `fwi-global-forecast-latest.png` | `2160x1080` | Global Fire Weather Index Forecast (Nov 2026) -- NASA GFWED | `FWI Risk` |
| 6 | `airs-global-co2-60s.png` | `720x480` | AIRS Global Tropospheric CO2 (365-435 ppm) -- NASA Aqua / AIRS | `ppm CO2` |
| 7 | `svs-co2-column-2021-05-16.png` | `1860x935` | NASA SVS 3D Carbon Dioxide Column (May 2021) -- NASA SVS | `ppm CO2` |

*(Note: Dataset order in the UI follows sorted filename order plus the tactile reference map as dataset `8`.)*

---

## Dataset Details & Color-to-Science Mappings

### 1. Global PM2.5 Aerosol Optical Depth (2019)
- **File**: `data/images/aerosol-optical-depth-pm25-2019.png`
- **Category**: Air Quality & Aerosols (MODIS / MISR / SeaWiFS)
- **Scientific Summary**: Tracks global fine particulate matter (PM2.5) concentrations from `0` to `139 ug/m3`.
- **Color Meanings**:
  - `Red`: Severe fine particulate hotspot (`40` to `139 ug/m3`)
  - `Orange`: Elevated dust or smoke particulate load (`10` to `80 ug/m3`)
  - `Blue`: Clean air corridor or ocean zone (`< 10 ug/m3`)
  - `White`: Polar land mask or chart margin

### 2. GEOS Global Aerosol Transport (Aug 2025)
- **File**: `data/images/geos-aerosols-2025-08-19.png`
- **Category**: Atmospheric Transport (NASA GEOS)
- **Scientific Summary**: Global atmospheric aerosol transport simulation tracing marine sea salt, Saharan mineral dust, and biomass burning smoke.
- **Color Meanings**:
  - `Red`: Intense wildfire smoke or concentrated biomass burning emission
  - `Orange`: Saharan mineral dust plume (including violet dust plumes mapped to `Orange`) or warm smoke transport corridor
  - `Green`: Marine sea-salt aerosol lifted by ocean storm winds
  - `Blue`: Cyclonic sea-spray swirl or temperate marine aerosol
  - `Black`: Calm background atmosphere with minimal aerosol load

### 3. ECCO2 Perpetual Ocean Currents & Temperature
- **File**: `data/images/ecco2-ocean-currents-grid.jpg`
- **Category**: Ocean Circulation & Climate (NASA ECCO2)
- **Scientific Summary**: Ultra-high-resolution (`6840x3420` native, scaled to `960x480` at load) global ocean circulation model tracing surface currents and eddies across thermal bands.
- **Color Meanings**:
  - `Red`: Warmest tropical ocean current or intense boundary eddy
  - `Orange`: Warm equatorial ocean current or subtropical gyre
  - `Green`: Temperate mid-latitude current stream and upwelling belt
  - `Blue`: Cold polar and subpolar ocean current
  - `Black`: Continental landmass or calm abyssal basin

### 4. North America Fire Weather Index (Aug 2026)
- **File**: `data/images/fwi-conus-2026-08-01.png`
- **Category**: Wildfire Extremes (NASA GFWED)
- **Scientific Summary**: Regional Fire Weather Index for August 1, 2026 across North America, highlighting extreme wildfire weather risk in the western interior.
- **Color Meanings**:
  - `Red`: Very high to extreme fire weather danger (including deep crimson cores)
  - `Orange`: Low to moderate fire weather danger
  - `Green`: Very low fire weather danger with moist conditions
  - `Black`: Ocean water or unvegetated high-altitude mask
  - `Blue`: Legend box or coastal water boundary

### 5. Global Fire Weather Index Forecast (Nov 2026)
- **File**: `data/images/fwi-global-forecast-latest.png`
- **Category**: Global Wildfire Forecast (NASA GFWED)
- **Scientific Summary**: Global Fire Weather Index forecast across continents, highlighting active fire weather belts across Africa and Australia.
- **Color Meanings**:
  - `Red`: Very high to extreme global fire weather risk belt
  - `Orange`: Low to moderate fire weather risk
  - `Green`: Very low fire weather risk
  - `White`: Unvegetated desert, polar snow, or ocean background mask
  - `Black`: Extreme wildfire core or polar shoreline

### 6. AIRS Global Tropospheric CO2 (365-435 ppm)
- **File**: `data/images/airs-global-co2-60s.png`
- **Category**: Greenhouse Gases (NASA Aqua / AIRS)
- **Scientific Summary**: Mid-tropospheric carbon dioxide concentrations measured by NASA's AIRS instrument (`365` to `435 ppm`).
- **Color Meanings**:
  - `Red`: Peak regional CO2 concentration approaching `435 ppm`
  - `Orange`: Elevated tropospheric CO2 around `395` to `415 ppm`
  - `Black`: Polar frame border or legend banner
  - `White`: Baseline `365 ppm` scale floor

### 7. NASA SVS 3D Carbon Dioxide Column (May 2021)
- **File**: `data/images/svs-co2-column-2021-05-16.png`
- **Category**: Atmospheric Carbon Plumes (NASA SVS)
- **Scientific Summary**: NASA Scientific Visualization Studio 3D atmospheric carbon dioxide column visualization (`385` to `425 ppm`, global mean `416.8 ppm`).
- **Color Meanings**:
  - `Red`: Dense CO2 filament approaching `425 ppm`
  - `Orange`: Elevated northern hemisphere CO2 plume above `405 ppm`
  - `White`: Antarctic ice sheet or baseline scale band
  - `Black`: Lower southern hemisphere CO2 corridor over dark ocean

---

## 3D-Printed Tactile Map Hardware (`hardware/tactile-map/`)

The `hardware/tactile-map/` directory provides the physical relief models and matching reference projection used when exploring Earth Jukebox on a Linux trackpad:

| File | Size | Dimensions | Purpose |
| :--- | :--- | :--- | :--- |
| `hardware/tactile-map/world-map.3mf` | ~2.05 MB | 3D Model | Primary 3D-printable project file (3MF) for the tactile world map overlay |
| `hardware/tactile-map/world-map.stl` | ~8.65 MB | 3D Mesh | Standard STL triangle mesh for universal 3D printer slicers |
| `hardware/tactile-map/world-map.jpg` | ~18 KB | `630x350` | Reference world topography/bathymetry map aligned to the 3D tactile model (loaded as dataset `8`) |

### Alignment Notes

- The tactile overlay shares the global projection of `world-map.jpg` so tactile continental boundaries correspond to the global equirectangular NASA EIC maps (`geos-aerosols-2025-08-19.png`, `ecco2-ocean-currents-grid.jpg`, `fwi-global-forecast-latest.png`, `svs-co2-column-2021-05-16.png`).
- Binary 3D models and high-resolution raster assets (`*.stl`, `*.3mf`, `*.png`, `*.jpg`) are tracked with Git LFS rules in `.gitattributes`.
