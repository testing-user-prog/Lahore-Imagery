import ee
import requests
import os


def get_coverage_ratio(image, region):
    intersection = region.intersection(image.geometry(), ee.ErrorMargin(1))
    coverage_area = intersection.area(ee.ErrorMargin(1))
    region_area = region.area(ee.ErrorMargin(1))
    ratio = ee.Number(coverage_area).divide(region_area)
    return image.set('coverage_ratio', ratio)


def find_best_image(earth_enginemodel, geometry_points, start_date, end_date, cloud_prop, extrapolate=False):
    point = ee.Geometry.Point(geometry_points)
    region = point.buffer(2000).bounds()

    collection = (ee.ImageCollection(earth_enginemodel)
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt(cloud_prop, 20)))

    

    
    collection_with_coverage = collection.map(lambda img: get_coverage_ratio(img, region))
    well_covered = collection_with_coverage.filter(ee.Filter.gte('coverage_ratio', 0.95))
    final_collection = well_covered.sort(cloud_prop)

    count = final_collection.size().getInfo()
    print(f"Found {count} fully-covering, low-cloud images")

    
    return final_collection.first(), region


def find_composite_image(earth_enginemodel, geometry_points, start_date, end_date, cloud_prop):
    point = ee.Geometry.Point(geometry_points)
    region = point.buffer(2000).bounds()

    collection = (ee.ImageCollection(earth_enginemodel)
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt(cloud_prop, 30)))

    count = collection.size().getInfo()
    print(f"Found {count} images to build gap-filled composite from")

    return collection.median(), region


def download_image(image, region, bands, min_val, max_val, imagesavepath, imagename):
    thumb_url = image.getThumbURL({
        'region': region,
        'dimensions': 512,
        'bands': bands,
        'min': min_val,
        'max': max_val
    })
    print("Preview image URL:", thumb_url)

    os.makedirs(imagesavepath, exist_ok=True)
    response = requests.get(thumb_url)
    if response.status_code != 200:
        print(f"Failed to download image. Status code: {response.status_code}")
        return False

    file_path = os.path.join(imagesavepath, imagename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Image saved to: {file_path}")
    return True


def imagefromearthengine(imagesavepath, imagename, start_date, end_date, earth_enginemodel, geometry_points, cloud_prop, date_field, bands, min_val, max_val,extrapolate=False):
    if extrapolate==False:
        image, region = find_best_image(earth_enginemodel, geometry_points, start_date, end_date, cloud_prop)
    else:
        image, region = find_composite_image(earth_enginemodel, geometry_points, start_date, end_date, cloud_prop)

    if image is None:
        print("No fully-covering, low-cloud images found for this date range.")
        return False

    info = image.getInfo()
    print("Image ID:", info.get('id', 'Composite (no single ID)'))
    # Uses whichever field name was passed in for this sensor - no sensor-specific
    # knowledge lives inside this function anymore.
    acquisition_date = info.get('properties', {}).get(date_field, 'Unknown')
    print("Date:", acquisition_date)
    print("Cloud %:", info.get('properties', {}).get(cloud_prop, 'N/A'))
    print("Coverage ratio:", info.get('properties', {}).get('coverage_ratio', 'N/A'))

    return download_image(image, region, bands, min_val, max_val, imagesavepath, imagename)