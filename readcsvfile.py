import csv
import scipy.stats
from pylab import *
import numpy
from scanner_lifespans import run_analysis

"workflow:"
"import readcsvfile as r"
"raw_lifespans=r.read_file('/Users/zplab/Desktop/2015_03_03_fluorescencelifespandata/2015-03-03-assay-evaluated_lifespans.csv', 'lifespan', true)" 
"raw_fluorescence_median_values=r.readfile('/Users/zplab/Desktop/2015_03_03_fluorescencelifespandata/2015-03-03-fluorescencevalues.csv', 'median', false)"
"lifespans, fluorescence = r.delete_zeros(raw_lifespans, raw_fluorescence_median_values)"
"r.run_stats(lifespans, fluorescence)"


"read_file opens a csv file. Need to have headers to work. Give it the file_name(drag and drop), header_name"
"as string. lifespan_file is true if the file contains lifespan data, false otherwise"

wellies = 'BCDEFGHIJKLMNO'



def process_image_dir_auto(in_dir, age_at_first_scan):
	out_dir = in_dir + " OUT"
	run_analysis.process_image_dir(in_dir, out_dir, age_at_first_scan, run_analysis.HOLLY_NAME_PARAMS, run_analysis.HOLLY_PLATE_PARAMS, run_analysis.HOLLY_IMAGE_SCORE_PARAMS)

def find_col_lifespans(lifespans, wells):
	sorted_lifespans =[]

	for i in range(0,22):
		sorted_lifespans.append([])


	for i in range(0, 22):
		for j, well in enumerate(wells):
			if ((int)(well[1]+well[2])-2)==i:
				sorted_lifespans[i].append(lifespans[j])	
	return(sorted_lifespans)
				
def find_row_lifespans(lifespans, wells):
	sorted_lifespans=[]
		
	for letter in 'BCDEFGHIJKLMNO':
		sorted_lifespans.append([])
	for i in range(0, len(wellies)):
		for j, well in enumerate(wells):
			if well[0]==wellies[i]:
				sorted_lifespans[i].append(lifespans[j])
					

	return (sorted_lifespans)	


def plot_row_means_and_medians(sorted_lifespans, rows, out_dir):
	
	row_list=[]
	means=[]
	medians=[]
	

	for i in range(0, len(sorted_lifespans)):
		row_list.append([])
		means.append(sum(sorted_lifespans[i])/len(sorted_lifespans[i]))	
		medians.append(numpy.median(sorted_lifespans[i]))
	
	for i in range(0,len(sorted_lifespans)):
		for j in range(0, len(sorted_lifespans[i])):
			row_list[i].append(i)	

	means_row_list=arange(0,len(means))		
	means_row_list=numpy.asarray(means_row_list)
	means=numpy.asarray(means)
	medians=numpy.asarray(medians)		
	
	means_correlation=scipy.stats.pearsonr(means_row_list, means)
	means_correlation_spearman=scipy.stats.spearmanr(means_row_list, means)
	means_correlation_spearman=numpy.asarray(means_correlation_spearman)
	means_correlation = numpy.asarray(means_correlation)
	(m,b) = polyfit(means_row_list, means, 1)
	yp=polyval([m,b], means_row_list)

	medians_correlation=scipy.stats.pearsonr(means_row_list, medians)
	medians_correlation_spearman=scipy.stats.pearsonr(means_row_list, medians)
	medians_correlation_spearman=numpy.asarray(medians_correlation_spearman)
	medians_correlation = numpy.asarray(medians_correlation)
	(mm,bb) = polyfit(means_row_list, medians, 1)
	ypp=polyval([mm,bb], means_row_list)

	for i in range(0, len(sorted_lifespans)):
		plt.scatter(row_list[i], sorted_lifespans[i])
	
	if rows:
		title="Row Number vs. Evaluated Lifespan"
		plt.title(title)
		plt.xlabel("Row Number")
	else:
		title="Column Number vs Evaluated Lifespan"
		plt.title(title)
		plt.xlabel("Column Number")
	plt.ylabel("Lifespan (Days)")
	plt.gcf()
	plt.savefig(out_dir+'/'+title+'.png')
	plt.show()
	plt.gcf().clf


	plot(means_row_list,yp)
	plt.scatter(means_row_list, means)
	if rows:
		title="Row Number vs. Mean Evaluated Lifespan"
		plt.title(title)
		plt.xlabel("Row Number")
	else:
		title=" Column Number vs. Mean Evaluated Lifespan"
		plt.title(title)
		plt.xlabel("Column Number")
	plt.ylabel("Lifespan (Days)")
	ftext="r^2="+(str)((means_correlation[0])**2)+" "+"p="+(str)(means_correlation[1])
	gtext="rho="+(str)((means_correlation_spearman[0]))+" "+"p="+(str)(means_correlation_spearman[1])
	plt.figtext(.5,.8,ftext,fontsize=11,ha='left')
	plt.figtext(.5,.7,gtext,fontsize=11,ha='left')

	plt.gcf()
	plt.savefig(out_dir+'/' +title+'.png')
	plt.show()
	plt.gcf().clf
	
	plot(means_row_list,ypp)
	plt.scatter(means_row_list, medians)
	if rows:
		title="Row Number vs. Median Evaluated Lifespan"
		plt.title(title)
		plt.xlabel("Row Number")
	else:
		title="Column Number vs. Median Evaluated Lifespan"
		plt.title(title)
		plt.xlabel("Column Number")
	plt.ylabel("Lifespan (Days)")
	ftext="r^2="+(str)((medians_correlation[0])**2)+" "+"p="+(str)(medians_correlation[1])
	gtext="rho="+(str)((medians_correlation_spearman[0]))+" "+"p="+(str)(medians_correlation_spearman[1])
	plt.figtext(.5,.8,ftext,fontsize=11,ha='left')
	plt.figtext(.5,.7,gtext,fontsize=11,ha='left')
	plt.gcf()
	plt.savefig(out_dir+'/'+title+'.png')
	plt.show()
	plt.gcf().clf	

def living_vs_dead_scores(scored_dir, total_dictionary, lifespan_list):
	data=run_analysis.load_data(scored_dir)
	scores=data.scores
	ages=data.ages
	well_names=data.well_names

	score_dictionary=dict.fromkeys(well_names)

	for well_name in well_names:
		score_dictionary[well_name]=[]

	for index, well_name in enumerate(well_names):
		for j in range(0,len(ages)):
			score_dictionary[well_name].append(scores[index][j])

	new_score_dictionary={}

	for key in score_dictionary:
		entry=score_dictionary[key]
		if key in total_dictionary:
			new_score_dictionary[key]=entry

	last_alive_dictionary=dict.fromkeys(new_score_dictionary)
	

	for key in new_score_dictionary:
		entry=total_dictionary[key]
		for i in range(1, len(ages)):
			if entry[0]==((ages[i]+ages[i-1])/2) or entry[0]==ages[i]:
				last_alive_dictionary[key]=i

	

	dead_scores=[]
	live_scores=[]

	for key in new_score_dictionary:
		score_entry=new_score_dictionary[key]
		for index, item in enumerate(score_entry):
			if index <= last_alive_dictionary[key]:
				live_scores.append(item)
			else:
				dead_scores.append(item)	
	plt.hist(live_scores)
	plt.hist(dead_scores)
	plt.title("Histogram of Scores from Dead and Living Worms")
	plt.xlabel("Scores")
	plt.ylabel("Count")
	plt.show()	
	return dead_scores, live_scores		

def read_estimated_lifespans(estimated_lifespan_file_name, dictionary, scored_dir):
	estimated_lifespans=[]
	well_names=[]

	data=run_analysis.load_data(scored_dir)
	ages=data.ages

	with open(estimated_lifespan_file_name) as csvfile:
		reader=csv.DictReader(csvfile)
		for row in reader:
			estimated_lifespans.append(row[('lifespan')])
			well_names.append(row[('well name')])

	estimated_lifespans_float=[]

	for index in range (0, len(well_names)):
		if estimated_lifespans[index]=='nan':
			estimated_lifespans_float.append(ages[-1])
		else:	
			estimated_lifespans_float.append((float)(estimated_lifespans[index]))
		


	estimated_dictionary=dict.fromkeys(well_names)

	for index, well_name in enumerate (well_names):

		estimated_dictionary[well_name]=estimated_lifespans_float[index]

	new_estimated_dictionary={}	
	for key in dictionary:
		entry=estimated_dictionary[key]
		new_estimated_dictionary[key]=entry


	new_estimated_lifespan_list=[]

	for key, entry in sorted (new_estimated_dictionary.items()):
		entry=estimated_dictionary[key]
		new_estimated_lifespan_list.append(entry)

	return new_estimated_lifespan_list				

def read_all_files(lifespan_file_name, fluorescence_file_name, bad_wells, scored_dir):
	well_names=[]
	most_well_names=[]
	lifespans=[]
	integrated=[]
	percentile=[]
	median=[]
	area=[]
	expression_area=[]
	expression_area_fraction=[]
	expression_mean=[]
	high_expression_area=[]
	high_expression_area_fraction=[]
	high_expression_mean=[]
	high_expression_integrated=[]

	data=run_analysis.load_data(scored_dir)
	ages=data.ages
	
	with open(lifespan_file_name) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			lifespans.append(row[('lifespan')])
			most_well_names.append(row[('well name')])

	lifespan_dictionary = dict.fromkeys(most_well_names)	


	with open(fluorescence_file_name) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			integrated.append(row[('integrated')])
			median.append(row[('median')])	
			percentile.append(row[('percentile95')])
			area.append(row[('area')])
			well_names.append(row[('well')])
			expression_area.append(row[('expression_area')])
			expression_area_fraction.append(row[('expression_area_fraction')])
			expression_mean.append(row[('expression_mean')])
			high_expression_area.append(row[('high_expression_area')])
			high_expression_area_fraction.append(row[('high_expression_area_fraction')])
			high_expression_mean.append(row[('high_expression_mean')])
			high_expression_integrated.append(row[('high_expression_integrated')])

	
	dictionary = dict.fromkeys(well_names)	

	for well_name in well_names:
		dictionary[well_name]=[]		
	
	lifespans_float=[]
	integrated_float=[]
	percentile_float=[]
	median_float=[]
	area_float=[]
	expression_area_float=[]
	expression_area_fraction_float=[]
	expression_mean_float=[]
	high_expression_area_float=[]
	high_expression_area_fraction_float=[]
	high_expression_mean_float=[]
	high_expression_integrated_float=[]

	for index in range(0, len(most_well_names)):
		if lifespans[index]=='nan':
			lifespans_float.append(ages[-1])
		else:	
			lifespans_float.append((float)(lifespans[index]))
	for index in range(0, len(well_names)):		
		integrated_float.append((float)(integrated[index]))
		percentile_float.append((float)(percentile[index]))
		median_float.append((float)(median[index]))
		area_float.append((float)(area[index]))
		expression_area_float.append((float)(expression_area[index]))
		expression_area_fraction_float.append((float)(expression_area_fraction[index]))
		expression_mean_float.append((float)(expression_mean[index]))
		high_expression_area_float.append((float)(high_expression_area[index]))
		high_expression_mean_float.append((float)(high_expression_mean[index]))
		high_expression_integrated_float.append((float)(high_expression_integrated[index]))

		

	


	for index, well_name in enumerate(most_well_names):
		if well_name in dictionary:
			dictionary[well_name].append(lifespans_float[index])
	for index, well_name in enumerate(well_names):
		if well_name in dictionary:	
			dictionary[well_name].append(integrated_float[index])
			dictionary[well_name].append(percentile_float[index])
			dictionary[well_name].append(median_float[index])
			dictionary[well_name].append(area_float[index])
			dictionary[well_name].append(expression_area_float[index])
			dictionary[well_name].append(expression_area_fraction_float[index])
			dictionary[well_name].append(expression_mean_float[index])
			dictionary[well_name].append(high_expression_area_float[index])
			dictionary[well_name].append(high_expression_mean_float[index])
			dictionary[well_name].append(high_expression_integrated_float[index])

	for well in bad_wells:
		if well in dictionary:
			del dictionary[well]

	new_dictionary={}		
	
	for key in dictionary:
		entry=dictionary[key]
		if entry[0]>0.0:
			new_dictionary[key]=entry
	return new_dictionary		
	
def make_lists(new_dictionary, manual_outliers):		

	if manual_outliers != '':
		for item in manual_outliers:
			new_dictionary.pop(item)
	lifespan_list=[]
	integrated_list=[]
	area_list=[]
	median_list=[]
	percentile_list=[]
	expression_area_list=[]
	expression_area_fraction_list=[]
	expression_mean_list=[]
	high_expression_area_list=[]
	high_expression_mean_list=[]
	high_expression_integrated_list=[]
	well_names=[]


	for key, entry in sorted(new_dictionary.items()):
		well_names.append(key)
		lifespan_list.append(entry[0])
		integrated_list.append(entry[1])
		percentile_list.append(entry[2])
		median_list.append(entry[3])
		area_list.append(entry[4])
		expression_area_list.append(entry[5])
		expression_area_fraction_list.append(entry[6])
		expression_mean_list.append(entry[7])
		high_expression_area_list.append(entry[8])
		high_expression_mean_list.append(entry[9])
		high_expression_integrated_list.append(entry[10])

	return new_dictionary, lifespan_list, integrated_list, percentile_list, median_list, area_list, well_names, expression_area_list, expression_area_fraction_list, expression_mean_list, high_expression_integrated_list, high_expression_mean_list, high_expression_area_list		



def make_lifespan_curve(lifespan_list, title, x_label, y_label, out_dir):
	
	total_worms=len(lifespan_list)

	lifespans=numpy.asarray(lifespan_list)

	max=lifespans.max()
	min=lifespans.min()

	if max is not int:
		max=(int)(max+.5)

	if min is not int:
		min=(int)(min-.5)	

	percent_alive = []
	days=range(min, max)

	for i in days:
		sum =0
		for item in lifespans:
			if item > i:
				sum=sum+1
		percent_alive.append((sum/total_worms)*100)		

	plot(days,percent_alive)
	#plt.scatter(x_list, y_list)
	plt.xlim(xmin=(min-3))
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	ftext="mean lifespan="+(str)(lifespans.mean()) + " days"
	even_more_text="median lifespan = " + (str)(numpy.percentile(lifespans,50)) + " days"
	more_text="n= " + (str)(lifespans.size)
	plt.figtext(.5,.8,ftext,fontsize=11,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=11, ha='left')
	plt.figtext(.5, .75, even_more_text, fontsize=11, ha='left')
	plt.gcf()
	plt.savefig(out_dir+'/'+title+'.png')
	plt.show()
	plt.gcf().clf
	

	


def run_stats(x_list, y_list, title, x_label, y_label, out_dir):
	correlation=scipy.stats.pearsonr(x_list, y_list)
	correlation = numpy.asarray(correlation)
	spearman_correlation=scipy.stats.spearmanr(x_list, y_list)
	spearman_correlation=numpy.asarray(spearman_correlation)
	(m,b) = polyfit(x_list, y_list, 1)
	yp=polyval([m,b], x_list)
	plot(x_list,yp)
	plt.scatter(x_list, y_list)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if correlation[1]<.00001:
		p="p<.00001"
	else:
		p="p=" + ''+ (str)(round(correlation[1],5))
	if spearman_correlation[1]<.00001:
		spearman_p="p<.00001"
	else:
		spearman_p="p=" + '' + (str)(round(spearman_correlation[1],5))
				
	ftext="r^2="+(str)(round(correlation[0],2)**2)+" "+p
	#etext="r="+(str)(round(correlation[0],2))+" "+"p="+(str)(round(correlation[1],5))
	gtext="rho="+(str)(round(spearman_correlation[0],2))+" "+spearman_p
	more_text="n= " + (str)(len(x_list))
	plt.figtext(.5,.88,ftext,fontsize=11,ha='left')
	plt.figtext(.15,.8,gtext,fontsize=11,ha='left')
	#plt.figtext(.7,.88,etext,fontsize=11,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=11, ha='left')
	plt.gcf()
	plt.savefig(out_dir+'/'+title+'.png')
	plt.show()
	plt.gcf().clf
	
