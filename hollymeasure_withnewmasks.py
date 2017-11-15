import wormPhysiology.measurePhysiology.organizeData as od
from zplib.scalar_stats import mcd
import numpy as np
import scipy
import pandas as pd
import pathlib
import os
import time
import wormPhysiology.measurePhysiology.extractFeatures as extractFeatures
import wormPhysiology.wormFinding.backgroundSubtraction as backgroundSubtraction
import wormPhysiology.basicOperations.imageOperations as imageOperations
import freeimage


class HollyMeasurer(od.WormMeasurer):
	def __init__(self, worm_subdirectory, working_directory, validated_directory):
		print('Measuring the worm recorded in ' + worm_subdirectory + '...', flush = True)	
		super().__init__(worm_subdirectory, working_directory, validated_directory)
		self.worm_frame = pd.DataFrame(index = self.worm_times, columns = [
			'intensity_50', 'intensity_60', 'intensity_70', 'intensity_80', 'intensity_90', 'intensity_95', 'intensity_100', 'integrated_50', 'integrated_60', 'integrated_70', 'integrated_80', 'integrated_90', 'integrated_95', 'integrated_0', 
			'age_texture', 'vulval_texture', 'egg_texture', 'life_texture', 'intensity_texture',
			'unstimulated_rate', 'stimulated_rate_a', 'stimulated_rate_b', 'bulk_movement', 'fine_movement',
			'total_size', 'aspect_ratio',
			'visible_area', 'visible_eggs', 'average_egg_size', 'single_eggs', 'great_lawn_area', 'great_lawn_max_diameter',
			'integrated_gfp', 'median_gfp', 'percentile95_gfp', 'expressionarea_gfp','expressionareafraction_gfp','expressionmean_gfp','highexpressionarea_gfp',
			'highexpressionareafraction_gfp', 'highexpressionmean_gfp', 'highexpressionintegrated_gfp'

		])	
	
	def make_measurements(self, my_time, last_mask, focal_mask, movement_masks,temporal_radius,i, last_frame, focal_frame, eggs_mask):
		'''
		Given the positions of the worm and eggs and raw data files, make the actual measurements that I want.
		'''
		focal_mask = freeimage.read(self.working_directory + os.path.sep + self.full_worm_name + os.path.sep + my_time + ' ' + 'mask.png').astype('bool')
		
		self.t0 = time.time()	
		j = self.timepoints.index(my_time)		
		
		movement_colored=focal_mask
		bulk_move_colored=focal_mask	
		# Measure size and set up last_mask.
		last_mask = focal_mask.copy()		
		last_frame = focal_frame.copy()
		self.worm_frame.loc[my_time] = extractFeatures.measure_size(focal_mask, self.worm_frame.loc[my_time])
		
		# Only do autofluorescence measurements if fluorescence was on for this time point.
		if os.path.isfile(self.worm_subdirectory + os.path.sep + my_time + ' ' + 'green_yellow_excitation_autofluorescence.png'):
			focal_fluorescence = self.read_corrected_fluorescence(my_time)
			(self.worm_frame.loc[my_time], fluorescence_colored) = extractFeatures.measure_autofluorescence(focal_fluorescence, focal_mask, self.worm_frame.loc[my_time])
		else:
			focal_fluorescence = np.zeros(focal_frame.shape).astype('uint8')
			fluorescence_colored = focal_fluorescence
		#print('\tMeasured movement, size, and fluorescence for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		#GFP measurements

		if os.path.isfile(self.worm_subdirectory + os.path.sep + my_time + ' ' + 'gfp.png'):
			gfp_focal_fluorescence = self.read_corrected_gfp_fluorescence(my_time)
			(self.worm_frame.loc[my_time], gfp_fluorescence_colored) = self.measure_gfp_fluorescence(gfp_focal_fluorescence, focal_mask, self.worm_frame.loc[my_time])
		else:
			gfp_focal_fluorescence = np.zeros(focal_frame.shape).astype('uint8')
			gfp_fluorescence_colored = gfp_focal_fluorescence
		print('\tMeasured gfp fluorescence for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)


		# Measure the various textures.
		self.t0 = time.time()			
		self.worm_frame.loc[my_time] = extractFeatures.measure_texture(focal_frame, focal_mask, [self.age_texture_codebook, self.egg_texture_codebook, self.ghost_texture_codebook], [self.age_texture_regressor, self.egg_texture_regressor, self.life_texture_regressor], self.worm_frame.loc[my_time])
		print('\tMeasured texture for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Measure the eggs.
		self.t0 = time.time()			
		(self.worm_frame.loc[my_time], worm_eggs_colored) = extractFeatures.measure_eggs(eggs_mask, focal_mask, self.worm_frame.loc[my_time])
		print('\tMeasured eggs for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		return (last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored, gfp_fluorescence_colored,gfp_focal_fluorescence)
	def measure_a_time(self, current_context, my_time, last_frame, last_mask, i, temporal_radius):
		'''
		Measure a time point for this worm. It is called by self.measure_a_worm().
		'''
		print(str(i+1) + '/' + str(self.max_legitimate_times) + ': Measuring ' + my_time + ', ' + self.full_worm_name + '.', flush = True)

		# Find the worm!!! This part is different between measure_one_time and measure_a_time.			
		self.t1 = time.time()
		self.t0 = time.time()
		focal_frame = self.read_corrected_bf(my_time)
		movement_frames = [self.read_corrected_bf(my_time, movement_key = movement_key) for movement_key in ['00']]
		(current_context, background_model, background_mask, movement_masks) = backgroundSubtraction.background_frame(current_context, focal_frame, movement_frames, self.bacterial_lawn, i) 		
		print('\tGot masks from background subtraction for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Find the worm!!! This part is the same for both measure_one_time and measure_a_time.
		#focal_mask = freeimage.read(self.working_directory + os.path.sep + self.full_worm_name + os.path.sep + my_time + ' ' + 'mask.png').astype('bool')
		#eggs_mask=freeimage.read(self.working_directory + os.path.sep + self.full_worm_name + os.path.sep + my_time + ' ' + 'mask.png').astype('bool')
		(focal_mask, eggs_mask) = self.mask_decision(my_time, focal_frame, background_mask)
		print('\tGot final masks for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t1)[:5] + ' seconds.', flush = True)

		# Make all my measurements.
		(last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored,gfp_fluorescence_colored,focal_gfp_fluorescence) = self.make_measurements(my_time, last_mask, focal_mask, movement_masks, temporal_radius, i, last_frame, focal_frame, eggs_mask)		
		self.last_size = min(self.worm_frame.loc[my_time, 'total_size'], 100000)
		
		# Write out my results.
		self.t0 = time.time()	
		self.write_results(my_time, focal_frame, background_model, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored,gfp_fluorescence_colored,focal_gfp_fluorescence)
		print('\tWrote results for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)		
		return (current_context, last_frame, last_mask)
	def write_results(self, my_time, focal_frame, background_frame, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored,gfp_fluorescence_colored,focal_gfp_fluorescence, write_health = True):
		'''
		Write out my results to disk as I go. This is how things should go!
		'''
		# Prepare directories and a base name to write to for the time point.
		os.makedirs(self.working_directory, exist_ok = True)
		os.makedirs(self.write_directory, exist_ok = True)
		base_name = self.write_directory + os.path.sep + my_time + ' '

		# Write out my images.
		#freeimage.write(focal_frame.astype('uint16'), base_name + 'bf.png')
		#freeimage.write(imageOperations.renormalize_image(focal_mask.astype('uint8')), base_name + 'mask.png')
		#freeimage.write(imageOperations.renormalize_image(focal_fluorescence.astype('uint16')), base_name + 'autofluorescence.png')
		#freeimage.write(imageOperations.renormalize_image(focal_gfp_fluorescence.astype('uint16')),base_name + 'gfp.png')
		#freeimage.write(background_frame, base_name + 'background.png')
		#freeimage.write(worm_eggs_colored, base_name + 'color_worm_eggs.png')
		freeimage.write(fluorescence_colored, base_name + 'color_autofluorescence.png')
		#freeimage.write(movement_colored, base_name + 'color_movement.png')
		#freeimage.write(bulk_move_colored, base_name + 'color_bulk_movement.png')
		freeimage.write(gfp_fluorescence_colored,base_name+'color_gfp_fluorescence.png')
		
		# Write out the measured data.
		if write_health:
			final_directory = self.experiment_directory + os.path.sep + 'measured_health'
			os.makedirs(final_directory, exist_ok = True)
			self.worm_frame.to_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t')
		return	
	def measure_gfp_fluorescence(self,fluorescent_image, worm_mask, time_series):
		worm_pixels = fluorescent_image[worm_mask].copy()

		low_px_mean, low_px_std = mcd.robust_mean_std(worm_pixels[worm_pixels < worm_pixels.mean()], 0.5)
		expression_thresh = low_px_mean + 2.5*low_px_std
		high_expression_thresh = low_px_mean + 6*low_px_std
		fluo_px = worm_pixels[worm_pixels > expression_thresh]
		high_fluo_px = worm_pixels[worm_pixels > high_expression_thresh]
		area = worm_mask.sum()
		integrated = worm_pixels.sum()
		median, percentile95 = np.percentile(worm_pixels, [50, 95])
		expression_area = fluo_px.size
		expression_area_fraction = expression_area / area
		expression_mean = fluo_px.mean()
		high_expression_area = high_fluo_px.size
		high_expression_area_fraction = high_expression_area / area
		high_expression_mean = high_fluo_px.mean()
		high_expression_integrated = high_fluo_px.sum()
		expression_mask = (fluorescent_image > expression_thresh) & worm_mask
		high_expression_mask = (fluorescent_image > high_expression_thresh) & worm_mask


		time_series.loc[['integrated_gfp', 'median_gfp', 'percentile95_gfp', 'expressionarea_gfp', 'expressionareafraction_gfp', 'expressionmean_gfp', 
		'highexpressionarea_gfp', 'highexpressionareafraction_gfp', 'highexpressionmean_gfp', 'highexpressionintegrated_gfp']] = (integrated, median, percentile95, expression_area, expression_area_fraction, expression_mean, high_expression_area, high_expression_area_fraction, high_expression_mean, high_expression_integrated)
	
		expression_area_mask = np.zeros(worm_mask.shape).astype('bool')
		high_expression_area_mask = np.zeros(worm_mask.shape).astype('bool')
		percentile95_mask = np.zeros(worm_mask.shape).astype('bool')
		
		expression_area_mask[fluorescent_image > expression_thresh] = True
		expression_area_mask[np.invert(worm_mask)] = False
		high_expression_area_mask[fluorescent_image > high_expression_thresh] = True
		high_expression_area_mask[np.invert(worm_mask)] = False
		percentile95_mask[fluorescent_image > percentile95] = True
		percentile95_mask[np.invert(worm_mask)] = False

		colored_areas = extractFeatures.color_features([expression_area_mask,high_expression_area_mask,percentile95_mask])	
		return (time_series, colored_areas)

	def read_corrected_gfp_fluorescence(self, time_point, hot_threshold = 10000):
		'''
		Correct fluorescence images for flatfield, and re-normalize to make the images more nicely viewable.	
		'''
		# Read in image and apply vignetting.
		image_path = self.worm_subdirectory + os.path.sep + time_point + ' ' + 'gfp.png'
		raw_image = freeimage.read(image_path)
		raw_image[np.invert(self.super_vignette)] = 0

		# Correct for flatfield.
		flatfield_path = self.calibration_directory + os.path.sep + time_point + ' ' + 'fl_flatfield.tiff'
		calibration_image = freeimage.read(flatfield_path)
		#corrected_image = raw_image
		corrected_image = raw_image*calibration_image	

		# Correct for hot pixels.
		median_image = scipy.ndimage.filters.median_filter(corrected_image, size = 3)
		difference_image = np.abs(corrected_image.astype('float64') - median_image.astype('float64')).astype('uint16')
		hot_pixels = difference_image > hot_threshold
		median_image_hot_pixels = median_image[hot_pixels]
		corrected_image[hot_pixels] = median_image_hot_pixels

		# Return the actual image.
		return corrected_image	
	