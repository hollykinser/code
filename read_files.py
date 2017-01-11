import csv
import scipy.stats
from pylab import *
import numpy
from scanner_lifespans import run_analysis
import pathlib



def read_evaluated_lifespans(evaluated_lifespan_filename, scored_dir):
	if pathlib.Path.exists(pathlib.Path(evaluated_lifespan_filename)):
		well_names=[]
		evaluated_lifespans=[]

		data=run_analysis.load_data(scored_dir)
		ages=data.ages
	

		with open(evaluated_lifespan_filename) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				evaluated_lifespans.append(row[('lifespan')])
				well_names.append(row[('well name')])
		
		keys=well_names
		values=evaluated_lifespans
	
		for index in range(0, len(values)):
			if values[index]=='nan':
				values[index]=(ages[-1])
	
		values=[float(i) for i in values]
		

		evaluated_lifespan_dictionary=dict(zip(keys,values))

		pop_list=[]
		
		for key, value in evaluated_lifespan_dictionary.items():
			if value < 0:
				pop_list.append(key)

		for i in range(0, len(pop_list)):
			evaluated_lifespan_dictionary.pop(pop_list[i])   	
		return evaluated_lifespan_dictionary
	else:
		return None
	
	

def read_lifespans(scored_dir, clean_up):
	
	evaluated_lifespan_filename=scored_dir+'/'+'evaluated_lifespans.csv'
	estimated_lifespan_filename=scored_dir+'/'+'lifespans.csv'
	evaluated_lifespan_dictionary=read_evaluated_lifespans(evaluated_lifespan_filename, scored_dir)

	well_names=[]
	estimated_lifespans=[]

	data=run_analysis.load_data(scored_dir)
	ages=data.ages
	

	with open(estimated_lifespan_filename) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			estimated_lifespans.append(row[('lifespan')])
			well_names.append(row[('well name')])

	keys=well_names
	values=estimated_lifespans
	
	for index in range(0, len(values)):
		if values[index]=='nan':
			values[index]=(ages[-1])
	
	values=[float(i) for i in values]
		

	estimated_lifespan_dictionary=dict(zip(keys,values))

	pop_list=[]
		
	for key, value in estimated_lifespan_dictionary.items():
		if value < 0:
			pop_list.append(key)

	for i in range(0, len(pop_list)):
		estimated_lifespan_dictionary.pop(pop_list[i]) 	
	if clean_up==True:
		estimated_lifespan_dictionary=read_files.clean_up_estimated_lifespan_dictionary(evaluated_lifespan_dictionary, estimated_lifespan_dictionary)		
 			
	return evaluated_lifespan_dictionary, estimated_lifespan_dictionary

def clean_up_estimated_lifespan_dictionary (evaluated_lifespan_dictionary, estimated_lifespan_dictionary):
	pop_list=[]
	data=run_analysis.load_data(scored_dir)
	ages=data.ages

	if evaluated_lifespan_dictionary !=None:
		for key, value in estimated_lifespan_dictionary.items():
			if evaluated_lifespan_dictionary.get(key)==None:
				pop_list.append(key) 	

		for i in range(0, len(pop_list)):
			estimated_lifespan_dictionary.pop(pop_list[i]) 

	return estimated_lifespan_dictionary		
			
def manually_delete_outliers (outliers, dictionary, scored_dir):
	text_file=open(scored_dir+'/Outliers.txt')
	for i in range(0, len(outliers)):
		dictionary.pop(outliers[i])
		text_file.write('\n'.join(outliers[i])
	text_file.close()
	
	return dictionary		



def read_fluorescence_files(fluorescence_file_name, bad_wells, scored_dir):
	well_names=[]
	lifespans=[]
	integrated=[]
	percentile=[]
	median=[]
	area=[]

	data=run_analysis.load_data(scored_dir)
	ages=data.ages


	with open(fluorescence_file_name) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			integrated.append(row[('integrated')])
			median.append(row[('median')])	
			percentile.append(row[('95th')])
			area.append(row[('area')])
			well_names.append(row[('well')])

	
	fluorescence_dictionary = dict.fromkeys(well_names)	

	for well_name in well_names:
		fluorescence_dictionary[well_name]=[]		
	
	lifespans_float=[]
	integrated_float=[]
	percentile_float=[]
	median_float=[]
	area_float=[]

	for index in range(0, len(well_names)):		
		integrated_float.append((float)(integrated[index]))
		percentile_float.append((float)(percentile[index]))
		median_float.append((float)(median[index]))
		area_float.append((float)(area[index]))		
	

	for index, well_name in enumerate(well_names):
			fluorescence_dictionary[well_name].append(integrated_float[index])
			fluorescence_dictionary[well_name].append(percentile_float[index])
			fluorescence_dictionary[well_name].append(median_float[index])
			fluorescence_dictionary[well_name].append(area_float[index])

	for well in bad_wells:
		if well in fluorescence_dictionary:
			del fluorescence_dictionary[well]

	return fluorescence_dictionary		
				
def compile_dictionary(fluorescence_dictionary, estimated_lifespan_dictionary, evaluated_lifespan_dictionary):
	
	if evaluated_lifespan_dictionary != None:
		for key, value in evaluated_lifespan_dictionary.items():
			if fluorescence_dictionary.get(key)==None:
				del fluorescence_dictionary[key]
			if estimated_lifespan_dictionary.get(key)==None:
				del estimated_lifespan_dictionary(key)	




def make_lists(new_dictionary, manual_outliers):
	if manual_outliers != '':
		for item in manual_outliers:
			new_dictionary.pop(item)
	lifespan_list=[]
	integrated_list=[]
	area_list=[]
	median_list=[]
	percentile_list=[]
	well_names=[]


	for key, entry in sorted(new_dictionary.items()):
		well_names.append(key)
		lifespan_list.append(entry[0])
		integrated_list.append(entry[1])
		percentile_list.append(entry[2])
		median_list.append(entry[3])
		area_list.append(entry[4])

	return new_dictionary, lifespan_list, integrated_list, percentile_list, median_list, area_list, well_names		



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
	ftext="r^2="+(str)(round(correlation[0],2)**2)+" "+"p="+(str)(round(correlation[1],5))
	etext="r="+(str)(round(correlation[0],2))+" "+"p="+(str)(round(correlation[1],5))
	gtext="rho="+(str)(round(spearman_correlation[0],2))+" "+"p="+(str)(round(spearman_correlation[1],5))
	more_text="n= " + (str)(len(x_list))
	plt.figtext(.5,.88,ftext,fontsize=11,ha='left')
	plt.figtext(.15,.8,gtext,fontsize=11,ha='left')
	plt.figtext(.7,.88,etext,fontsize=11,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=11, ha='left')
	plt.gcf()
	plt.savefig(out_dir+'/'+title+'.png')
	plt.show()
	plt.gcf().clf
	
