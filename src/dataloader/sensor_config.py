from src.dataloader.imagegenerator import imagefromearthengine


def get_sensor_settings(year, sensors_config):
    
    matching = [s for s in sensors_config if year >= s['min_year']]
    if not matching:
        raise ValueError(f"No sensor configuration found for year {year}")

    # Pick the one with the highest min_year among those that qualify
    return max(matching, key=lambda s: s['min_year'])


def saveimage(image_savepath, imagename, start_time, end_time, geometry_points, config):
    # Parse the year from start_time (which is in YYYY-MM-DD format)
    year = int(start_time.split('-')[0])

    sensor = get_sensor_settings(year, config['sensors'])

    return imagefromearthengine(
        image_savepath, imagename, start_time, end_time,
        sensor['earth_engine_model'], geometry_points,
        sensor['cloud_prop'], sensor['date_field'],
        sensor['bands'], sensor['min_val'], sensor['max_val'],
        sensor['extrapolate']
    )