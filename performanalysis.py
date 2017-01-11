import readcsvfile as r



def perform_all_analysis(out_dir, evaluated, fluorescence_file_name, autofluorescence, manual_outliers=''):
	
	if evaluated==True:
		evaluated_lifespans=out_dir+'/evaluated_lifespans.csv'
	else:
		evaluated_lifespans=out_dir+'/lifespans.csv'	
	
	bad_wells=['A23', 'A24', 'B23', 'B24']
	fluorescence_data=out_dir+"/"+fluorescence_file_name
	first_pass_dictionary=r.read_all_files(evaluated_lifespans, fluorescence_data, bad_wells, out_dir)
	dictionary, evaluated_lifespan_list, integrated_list, percentile_list, median_list, area_list, well_names, expression_area_list, expression_area_fraction_list, expression_mean_list, high_expression_integrated_list, high_expression_mean_list, high_expression_area_list=r.make_lists(first_pass_dictionary, manual_outliers)
	estimated_lifespans=out_dir+'/lifespans.csv'
	estimated_lifespan_list=r.read_estimated_lifespans(estimated_lifespans, dictionary, out_dir)
	

	if evaluated==True:
		r.run_stats(estimated_lifespan_list, evaluated_lifespan_list, "Estimated Lifespan vs. Evaluated Lifespan", "Estimated Lifespan(Days)", "Evaluated Lifespan (Days)", out_dir)
		r.make_lifespan_curve(evaluated_lifespan_list, "Survival Curve from Evaluated Lifespans", "Days", "Percent Survival", out_dir)
		r.run_stats(area_list, evaluated_lifespan_list, "Area vs. Evaluated Lifespan (Days)", "Area", "Evaluated Lifespan (Days)", out_dir)
		if autofluorescence==False:
			r.run_stats(percentile_list, evaluated_lifespan_list, "95th Percentile Fluorescence vs. Evaluated Lifespan", "95th Percentile Fluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(integrated_list, evaluated_lifespan_list, "Integrated Fluorescence vs. Evaluated Lifespan", "Integrated Fluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(median_list, evaluated_lifespan_list, "Median Fluorescence vs. Evaluated Lifespan", "Median Fluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_list, evaluated_lifespan_list, "Fluorescence Expression Area vs. Evaluated Lifespan", "Fluorescence Expression Area", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_fraction_list, evaluated_lifespan_list, "Fluorescence Expression Area Fraction vs. Evaluated Lifespan", "Fluorescence Expression Area Fraction", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_mean_list, evaluated_lifespan_list, "Fluorescence Expression Mean vs. Evaluated Lifespan", "Fluorescence Expression Mean", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_integrated_list, evaluated_lifespan_list, "Integrated High Fluorescence Expression vs. Evaluated Lifespan", "Integrated High Fluorescence Expression", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_mean_list, evaluated_lifespan_list, "Mean High Fluorescence Expression vs. Evaluated Lifespan", "Mean High Fluorescence Expression", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_area_list, evaluated_lifespan_list, "High Fluorescence Expression Area vs. Evaluated Lifespan", "High Fluorescence Expression Area", "Evaluated Lifespan (Days)", out_dir)
		else:
			r.run_stats(percentile_list, evaluated_lifespan_list, "95th Percentile Autofluorescence vs. Evaluated Lifespan", "95th Percentile Autofluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(integrated_list, evaluated_lifespan_list, "Integrated Autofluorescence vs. Evaluated Lifespan", "Integrated Autofluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(median_list, evaluated_lifespan_list, "Median Autofluorescence vs. Evaluated Lifespan", "Median Autofluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_list, evaluated_lifespan_list, "Autofluorescence Expression Area vs. Evaluated Lifespan", "Autofluorescence Expression Area", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_fraction_list, evaluated_lifespan_list, "Autofluorescence Expression Area Fraction vs. Evaluated Lifespan", "Autofluorescence Expression Area Fraction", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_mean_list, evaluated_lifespan_list, "Autofluorescence Expression Mean vs. Evaluated Lifespan", "Autofluorescence Expression Mean", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_integrated_list, evaluated_lifespan_list, "Integrated High Autofluorescence Expression vs. Evaluated Lifespan", "Integrated High Autofluorescence Expression", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_mean_list, evaluated_lifespan_list, "Mean High Autofluorescence Expression vs. Evaluated Lifespan", "Mean High Autofluorescence Expression", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_area_list, evaluated_lifespan_list, "High Autofluorescence Expression Area vs. Evaluated Lifespan", "High Autofluorescence Expression Area", "Evaluated Lifespan (Days)", out_dir)
		col_lifespans=r.find_col_lifespans(evaluated_lifespan_list, well_names)
		row_lifespans=r.find_row_lifespans(evaluated_lifespan_list, well_names)
		r.plot_row_means_and_medians(col_lifespans, False, out_dir)
		r.plot_row_means_and_medians(row_lifespans, True, out_dir)
			


	else:
		if autofluorescence==False:
			r.run_stats(area_list, estimated_lifespan_list, "Area vs. Estimated Lifespan (Days)", "Area", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(integrated_list, estimated_lifespan_list, "Integrated Fluorescence vs. Estimated Lifespan", "Integrated Fluorescence", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(median_list, estimated_lifespan_list, "Median Fluorescence vs. Estimated Lifespan", "Median Fluorescence", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(percentile_list, estimated_lifespan_list, "95th Percentile Fluorescence vs. Estimated Lifespan", "95th Percentile Fluorescence", "Estimated Lifespan (Days)", out_dir)	
			r.make_lifespan_curve(estimated_lifespan_list, "Survival Curve from Estimated Lifespans", "Days", "Percent Survival", out_dir)
			r.run_stats(expression_area_list, estimated_lifespan_list, "Fluorescence Expression Area vs. Estimated Lifespan", "Fluorescence Expression Area", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_fraction_list, estimated_lifespan_list, "Fluorescence Expression Area Fraction vs. Estimated Lifespan", "Fluorescence Expression Area Fraction", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(expression_mean_list, estimated_lifespan_list, "Fluorescence Expression Mean vs. Estimated Lifespan", "Fluorescence Expression Mean", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_integrated_list, estimated_lifespan_list, "Integrated High Fluorescence Expression vs. Estimated Lifespan", "Integrated High Fluorescence Expression", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_mean_list, estimated_lifespan_list, "Mean High Fluorescence Expression vs. Estimated Lifespan", "Mean High Fluorescence Expression", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_area_list, estimated_lifespan_list, "High Fluorescence Expression Area vs. Estimated Lifespan", "High Fluorescence Expression Area", "Estimated Lifespan (Days)", out_dir)
		else:
			r.run_stats(percentile_list, estimated_lifespan_list, "95th Percentile Autofluorescence vs. Estimated Lifespan", "95th Percentile Autofluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(integrated_list, estimated_lifespan_list, "Integrated Autofluorescence vs. Estimated Lifespan", "Integrated Autofluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(median_list, estimated_lifespan_list, "Median Autofluorescence vs. Estimated Lifespan", "Median Autofluorescence", "Evaluated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_list, estimated_lifespan_list, "Autofluorescence Expression Area vs. Estimated Lifespan", "Autofluorescence Expression Area", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(expression_area_fraction_list, estimated_lifespan_list, "Autofluorescence Expression Area Fraction vs. Estimated Lifespan", "Autofluorescence Expression Area Fraction", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(expression_mean_list, estimated_lifespan_list, "Autofluorescence Expression Mean vs. Estimated Lifespan", "Autofluorescence Expression Mean", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_integrated_list, estimated_lifespan_list, "Integrated High Autofluorescence Expression vs. Evaluated Lifespan", "Integrated High Autofluorescence Expression", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_mean_list, estimated_lifespan_list, "Mean High Autofluorescence Expression vs. Estimated Lifespan", "Mean High Autofluorescence Expression", "Estimated Lifespan (Days)", out_dir)
			r.run_stats(high_expression_area_list, estimated_lifespan_list, "High Autofluorescence Expression Area vs. Estimated Lifespan", "High Autofluorescence Expression Area", "Estimated Lifespan (Days)", out_dir)

	

	

	return dictionary