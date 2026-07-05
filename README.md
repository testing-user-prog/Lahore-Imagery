# Lahore Satellite Imagery Comparison

A pipeline for pulling satellite imagery of specific locations across multiple years, automatically selecting the correct sensor (Sentinel-2 or Landsat 5/7/8) based on the requested year, and saving clean, filtered images for visual and quantitative comparison over time.

## Features

- Automatically switches between Sentinel-2 and Landsat 5/7/8 depending on the requested year
- Filters candidate images by cloud cover and region coverage before selecting the best one
- Avoids Landsat 7's post-2003 Scan Line Corrector striping by routing that range to Landsat 5
- Optional gap-filled composite mode (`extrapolate`) for building multi-scene composites when a single clean image isn't available
- Sensor definitions (bands, thresholds, cloud property names) are stored in `config.json`, not hardcoded in the codebase

## Requirements

- Python 3.10+
- A Google Cloud project with the Earth Engine API enabled ([registration guide](https://developers.google.com/earth-engine/guides/access))

## Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Update `config.json` with your own Google Cloud project name:
   ```json
   {
     "project_name": "your-project-id",
     "save_path": "dataset",
     "sensors": [ ... ]
   }
   ```

3. Run the script:
   ```bash
   python app.py
   ```

   On first run, `ee.Authenticate()` will open a browser window asking you to log in and grant access to your Google account.

## Project Structure

```
├── app.py                        # Entry point - defines year range and coordinates, runs the loop
├── config.json                   # Project settings and per-sensor configuration
├── requirements.txt
├── src/
│   ├── authenticator/
│   │   └── authenticate.py       # One-time Earth Engine authentication
│   └── dataloader/
│       ├── sensor_config.py      # Picks the correct sensor settings for a given year
│       └── imagegenerator.py     # Searches for, filters, and downloads imagery
```

## Configuration

Each entry in `config.json`'s `sensors` list defines settings for one sensor:

| Field | Description |
|---|---|
| `min_year` | The earliest year this sensor applies to |
| `earth_engine_model` | The Earth Engine dataset ID |
| `cloud_prop` | The property name used for cloud percentage filtering |
| `date_field` | The property name used for the acquisition date |
| `bands` | Band names mapped to Red, Green, Blue for true-color display |
| `min_val` / `max_val` | Brightness stretch range for rendering |
| `extrapolate` | If `true`, builds a gap-filled composite instead of picking a single image (used for Landsat 7's SLC-off period) |

## Known Limitations

- Sentinel-2 coverage begins mid-2015; Landsat 5/7/8 cover earlier years
- Landsat 7 imagery after May 2003 has permanent diagonal data gaps due to a hardware failure; this pipeline avoids that range by default
- Coverage and cloud filtering may return no results for certain years/locations if no qualifying scene exists

## License

For personal/academic use.
