import numpy
import scipy.ndimage as ndimage
import freeimage
import re
import pathlib
import freeimage

def hysteresis_threshold(image, low_threshold, high_threshold):
    """Find all regions where image > low_threshold and which contain at least
    one pixel > high_threshold."""
    high_mask = image > high_threshold
    low_mask = image > low_threshold
    return ndimage.binary_dilation(high_mask, mask=low_mask, iterations=-1)

def remove_small_objects(mask, max_radius):
    """Remove objects from the mask up to max_radius (in terms of number of erosion
    iterations required to remove the object.)"""
    eroded = ndimage.binary_erosion(mask, iterations=max_radius)
    return ndimage.binary_dilation(eroded, mask=mask, iterations=-1)

def remove_edge_objects(mask):
    edge_objects = ndimage.binary_dilation(numpy.zeros_like(mask), mask=mask,
        border_value=1, iterations=-1)
    return binary_mask & ~edge_objects

def fill_small_holes(mask, max_radius):
    outside = ndimage.binary_dilation(numpy.zeros_like(mask), mask=~mask, iterations=-1, border_value=1)
    holes = ~(mask | outside)
    large_hole_centers = ndimage.binary_erosion(holes, iterations=max_radius)
    large_holes = ndimage.binary_dilation(large_hole_centers, mask=holes, iterations=-1)
    small_holes = holes ^ large_holes
    return mask | small_holes

def get_background(mask, offset_radius, background_radius):
    offset = ndimage.binary_dilation(mask, iterations=offset_radius)
    background = ndimage.binary_dilation(offset, iterations=background_radius)
    return background ^ offset

def get_areas(mask):
    labels, num_regions = ndimage.label(mask)
    region_indices = numpy.arange(1, num_regions + 1)
    areas = ndimage.sum(numpy.ones_like(mask), labels=labels, index=region_indices)
    return labels, region_indices, areas

def get_largest_object(mask):
    labels, region_indices, areas = get_areas(mask)
    largest = region_indices[areas.argmax()]
    return labels == largest

def remove_small_objects_exact(mask, max_area):
    labels, region_indices, areas = get_areas(mask)
    keep_labels = areas > max_area
    keep_labels = numpy.concatenate(([0], keep_labels))
    return keep_labels[labels]

def fill_small_holes_exact(mask, max_area):
    outside = ndimage.binary_dilation(numpy.zeros_like(mask), mask=~mask, iterations=-1, border_value=1)
    holes = ~(mask | outside)
    labels, region_indices, areas = get_areas(holes)
    keep_labels = areas <= max_area
    keep_labels = numpy.concatenate(([0], keep_labels))
    small_holes = keep_labels[labels]
    return mask | small_holes

def get_mask(image, low_pct, high_pct, max_hole_radius):
    low_thresh, high_thresh = numpy.percentile(image, [low_pct, high_pct])
    mask = hysteresis_threshold(image, low_thresh, high_thresh)
    mask = fill_small_holes(mask, max_hole_radius)
    mask = get_largest_object(mask)
    return mask

def measure_intensity(image):
    mask = get_mask(image, 99.2, 99.99, 12)
    background = get_background(mask, 15, 20)
    background_value = numpy.percentile(image[background], 25)
    pixel_data = image[mask] - background_value
    return (mask.sum(), pixel_data.sum())+ tuple(numpy.percentile(pixel_data, [50, 95]))

def measure_intensities(image_files, threads=3):
    import concurrent.futures as fut
    threadpool = fut.ThreadPoolExecutor(threads)
    def measure(image_file):
        print(str(image_file))
        image = freeimage.read(image_file)
        return measure_intensity(image)
    all_values = numpy.array(list(threadpool.map(measure, sorted(image_files))))
    return all_values

def get_well_names(image_files, well_regexp=r'([A-P])([0-9])([0-9])'):
    wells = []
    for image_file in sorted(image_files):
        m = re.match(well_regexp, image_file.name)
        well = m.group(1) + m.group(2)+m.group(3)
        wells.append(well)
    return wells

def write_intensities(all_values, image_files, csv_out):
    data_header = ['well', 'area', 'integrated', 'median', '95th']
    data = [data_header]
    wells = get_well_names(image_files)
    for well, values in zip(wells, all_values):
        data.append(map(str, [well] + list(values)))
    outdata = '\n'.join(','.join(row) for row in data)
    with open(csv_out, 'w') as f:
        f.write(outdata)
    return wells
    
def write_masks(image_files,image_dir):
    #image_files = sorted(list(image_dir.glob('*_fluorescence.tif')))
    wells = get_well_names(image_files)
    for image, well in zip(sorted(image_files), sorted(wells)):
        worm=freeimage.read(image)
        mask=get_mask(worm,99.2,99.99,12)
        mask=mask*255
        mask=mask.astype('uint8')
        freeimage.write(mask,(str)(image_dir)+'/'+well+'_mask.png')

def travis(image_dir):
    image_dir = pathlib.Path(image_dir)
    assert image_dir.exists()
    image_files = list(image_dir.glob('*_fluorescence.tif'))
    wells = get_well_names(image_files)
    image_data = measure_intensities(image_files)
    area = image_data[:,0]
    good_worms = area > 20000
    image_data = image_data[good_worms]
    integrated = image_data[:,1]
    wells = [w for w, g in zip(wells, good_worms) if g]
    num_to_take = int(round(len(wells) * 0.15))
    integ_order = integrated.argsort()
    high_wells = sorted([wells[i] for i in integ_order[-num_to_take:]])
    low_wells = sorted([wells[i] for i in integ_order[:num_to_take]])
    
    print('High GFP:')
    for w in high_wells:
        print(' '+w)
    print('Low GFP:')
    for w in low_wells:
        print(' '+w)
      
    #write_masks(image_files,image_dir)    