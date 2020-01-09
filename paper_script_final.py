from elegant import worm_data
from elegant import load_data
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from zplib.image import colorize
from zplib import datafile
import numpy
import scipy.stats
import time
import statsmodels.api as sm
from pandas import DataFrame
from sklearn import linear_model
from glob import glob
import re
import pathlib
from zplib import datafile
import freeimage
from elegant import worm_spline
import os
import pandas as pd
import seaborn
import zplib.image.mask as zpl_mask
import math


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = [15.0,9.0]
plt.rcParams['xtick.direction']='inout'
plt.rcParams['ytick.direction']='inout'
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
plt.rcParams['lines.dash_capstyle']='round'
plt.rcParams['lines.solid_capstyle']='round'
plt.rcParams['savefig.transparent']=True
plt.rcParams['legend.frameon']=False
plt.rcParams['svg.fonttype']='none'

plt.rcParams['axes.labelpad']=15.0
plt.rcParams['savefig.transparent']=True
plt.rcParams['legend.labelspacing']=.5
plt.rcParams['lines.linewidth']=3
plt.rcParams['xtick.labelsize']=22
plt.rcParams['ytick.labelsize']=22
plt.rcParams['xtick.major.pad'] = 9
plt.rcParams['ytick.major.pad'] = 9
plt.rcParams['lines.markersize']= 10

TIME_UNITS = dict(days=24, hours=1, minutes=1/60, seconds=1/3600)

label_fontdict={'size':20,'family':'arial'}
title_fontdict={'size':36, 'weight':'bold','family':'arial'}
text_fontdict={'size':22,'family':'arial'}
gfp_measures={
'gfp_mean':'mean intensity',
'gfp_median':'median intensity',
'gfp_maximum':'maximum intensity',
'gfp_over_99_mean':'mean over 99th percentile intensity',
'gfp_percentile_95':'95th percentile intensity',
'gfp_percentile_99':'99th percentile intensity',
'gfp_sum':'summed intensity',
'green_yellow_excitation_autofluorescence_percentile_95': '95th percentile intensity of autofluorescence'}


"""basic wrapper functions"""

def process_worms(paths,prefixes=['']):
	wormies=[]
	for path, prefix in zip(paths,prefixes):
		worms=worm_data.read_worms(path+'/*.tsv',name_prefix=prefix)
		wormies=wormies+worms
	
	return wormies

def get_average_wrapper(min_age,max_age,feature,age_feature='age'):
	def get_average(worm):
		data=worm.get_time_range(feature,min_age,max_age,age_feature=age_feature)
		return numpy.mean(data[1])
	return get_average

def get_slope_wrapper(min_age,max_age,feature,age_feature='age'):
	def get_slope(worm):
		data=worm.get_time_range(feature,min_age,max_age,age_feature=age_feature)
		slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(data)	
		return slope
	
	return get_slope

def get_peak_age_wrapper(min_age,max_age,feature,age_feature='age'):
	def get_peak_age(worm):
		data=worm.get_time_range(feature,min_age,max_age,age_feature=age_feature)
		peak_gfp=numpy.argmax(data[1])
		return data[0][peak_gfp]
	return get_peak_age	

def get_peak_wrapper(min_age,max_age,feature,age_feature='age'):
	def get_peak(worm):
		data=worm.get_time_range(feature,min_age,max_age,age_feature=age_feature)
		peak_gfp=numpy.max(data[1])
		return peak_gfp
	return get_peak

def bootstrap_confidence_interval(worms,n=10000):
	bootstrapped_correlations=[]
	for i in range(0,n):
		random_sample=numpy.random.choice(worms,len(worms),replace=True)			
		random_sample.regress()
		resid=random_pop[:,1]-random_pop[:,0]
		R2=1-(resid**2).mean(axis=0)/random_pop[:,1].var(axis=0)
		bootstrapped_correlations.append(R2)
	s_hat=numpy.mean(bootstrapped_correlations)
	low=numpy.percentile(bootstrapped_correlations,2.5)
	high=numpy.percentile(bootstrapped_correlations,97.5)
	return [2*s_hat-high,2*s_hat-low]	
def z_scoring_keypoint(subdirectories, age_feature='age',overwrite=True):
	#subdirectories=os.listdir(root_folder)
	#subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=[subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()]
		for exp_dir in exp_dirs:
			worms=process_worms([exp_dir], prefixes=[''])
			save_dir=pathlib.Path(exp_dir)
			if overwrite==False:
				continue
			for key in list(worms[0].td._keys()):
				if 'age' not in key and 'timepoint' not in key and 'timestamp' not in key:
	
					try:
						worms.z_transform(key,age_feature=age_feature)
					except:
						continue	
			print('done transforming '+exp_dir)
			worms.write_timecourse_data(save_dir)			


def z_scoring(root_folder, age_feature='age',gfp_measures=gfp_measures,overwrite=True):
	#z transform all files in a directory and save out file with the same name (will keep old measurements)
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=[subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()]
		for exp_dir in exp_dirs:
			worms=process_worms([exp_dir], prefixes=[''])
			save_dir=pathlib.Path(exp_dir)
			if overwrite==False:
				continue
			for key,value in gfp_measures.items():
	
				try:
					worms.z_transform(key,age_feature=age_feature)
				except:
					continue	
			print('done transforming '+exp_dir)
			worms.write_timecourse_data(save_dir)

#Supplementary Figure 1			
def make_fluor_worm_mask(root_folder):
	#makes worm mask colored by 95th, 99th, and 99.9 (approx. maximum) intensity percentile pixels
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir() and subdirectory[-1].isdigit()]
	choice=numpy.random.choice(subdirectories)
	miRNA=re.findall(r"\w\w\w-\d+",choice)[0]
	if choice[-3].isdigit():
		worm_number=choice[-3:]
	else:
		worm_number=choice[-2:]	
	paths_to_gfp_images=glob(choice+'/*gfp.png')
	paths_to_masks=glob(root_folder+'/derived_data/mask/'+worm_number+'/*bf.png')
	if not os.path.exists('/Volumes/9karray/Kinser_Holly/all_wild_type_fluor_masks/'+miRNA):
		os.mkdir('/Volumes/9karray/Kinser_Holly/all_wild_type_fluor_masks/'+miRNA)
	if not os.path.exists('/Volumes/9karray/Kinser_Holly/all_wild_type_fluor_masks/'+miRNA+'/'+worm_number+'/'):
		os.mkdir('/Volumes/9karray/Kinser_Holly/all_wild_type_fluor_masks/'+miRNA+'/'+worm_number+'/')	
	save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_fluor_masks/'+miRNA+'/'+worm_number+'/'
	
	for gfp_path, mask_path in zip(sorted(paths_to_gfp_images),sorted(paths_to_masks)):
		gfp_image=freeimage.read(gfp_path)
		print(gfp_path)
		mask_image=freeimage.read(mask_path)
		print(mask_path)
		mask=zpl_mask.get_largest_object(mask_image, structure=numpy.ones((3,3)))
		worm_pixels=gfp_image[mask]
		percentile_99=numpy.percentile(worm_pixels,99)
		percentile_95=numpy.percentile(worm_pixels,95)
		maximum=numpy.percentile(worm_pixels,99.9)
		mean=numpy.mean(worm_pixels)
		percentile_99_mask=(gfp_image>percentile_99)&mask
		percentile_95_mask=(gfp_image>percentile_95)&mask
		maximum_mask=(gfp_image>maximum)&mask
		mean_mask=(gfp_image>mean)&mask
		labels = numpy.zeros(mask.shape, dtype='uint16')
		labels[mask]=1
		labels[percentile_95_mask]=2
		labels[percentile_99_mask]=3
		labels[maximum_mask]=6
		color = colorize.colorize_label_image(labels)
		color[labels == 1] = 255 
		freeimage.write(color,save_name+mask_path[-22:-7]+'_color_mask.png')
		freeimage.write(gfp_image,save_name+mask_path[-22:-7]+'_gfp_image.png')
		freeimage.write(mask_image.astype('uint8'), save_name+mask_path[-22:-7]+'_mask.png')


"""wild type functions"""


#Figure 2A
def all_wild_type_lifespan_hist(root_folder):
	
	#plots histogram of raw lifespans with kde distribution
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		lifespans=worms.get_feature('lifespan')
		lifespans=pd.Series(lifespans/24)
		lifespans.plot.hist(bins=5,normed=True,color='gray',edgecolor='black',alpha=.2)
		lifespans.plot.kde()
		miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
		save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_lifespan_hist/'+miRNA+'_'+'.svg'
		save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_lifespan_hist/'+miRNA+'_'+'.svg'

		plt.title(miRNA+' lifespan distribution',fontdict=title_fontdict)
		plt.ylabel('Density', fontdict=label_fontdict,fontsize=24)
		plt.xlabel('Days',fontdict=label_fontdict,fontsize=24)
		print('Saving '+save_name)	
		plt.savefig(save_name)
		plt.savefig(save_name_box)
		plt.gcf().clf
		plt.close()

#Figure 1C
def all_wild_type_population_trajectory(root_folder,overwrite=True,min_age=24,time_units='days'):
	
	#plots population-level expression trace

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		max_age=numpy.inf
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory/'+miRNA+'_'+key+'.svg'

			try:
				worms.get_time_range(key,24,numpy.inf)
			except:
				continue	
			
			#rescales each experiment to median gfp and lifespan if there's more than one experiment directory
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,max_age)
						median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
						median_gfp[exp] = numpy.median([numpy.median(d[:,1]) for d in data])	
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
						gfp_feature='scaled_gfp'			
			else:
				lifespan_feature='lifespan'
				age_feature='age'
				gfp_feature=key

			lifespans=worms.get_feature(lifespan_feature)
			bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}
			averaged_worms = worm_data.meta_worms(bins, gfp_feature, age_feature=age_feature)
			figure=averaged_worms.plot_timecourse(gfp_feature,age_feature=age_feature,min_age=min_age,max_age=max_age,time_units=time_units)
			cohorts=[]
			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			text='n = ' + (str)(cohorts[0])
			plt.figtext(.8, .3, text, fontdict=text_fontdict, fontsize=24)
			plt.title('P'+miRNA+'::GFP expression',fontdict=title_fontdict, loc='left')
			ax = plt.gca() 
			ax.set_xlabel('Days', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.95, .05)
			ax.set_ylabel('Expression ('+value+')',fontdict=label_fontdict,fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=-.5,right=19)
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()		

#Figure 2D
def all_wild_type_population_trajectory_cohorts(root_folder,overwrite=True,min_age=24,time_units='days',rescale_lifespan=True):

	#generates cohort traces of average expression for each measure in gfp_measures
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		max_age=numpy.inf	
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
			lifespan_feature='lifespan'
			age_feature='age'	
		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			if rescale_lifespan:
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory_cohorts_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory_cohorts_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
			else:
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'	
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			try:
				worms.get_time_range(key,min_age,numpy.inf)
			except:
				continue	
			
			#rescales each experiment to median gfp, also lifespan and age if there's more than one experiment directory to facilitate pooling
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,numpy.inf)	
						median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
						median_gfp[exp] = numpy.median([numpy.median(d[:,1]) for d in data])
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						if rescale_lifespan:
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							age_feature='scaled_age'
							lifespan_feature='scaled_lifespan'
							gfp_feature='scaled_gfp'
						else:
							age_feature='age'
							lifespan_feature='lifespan'
							gfp_feature=key
			else:
				age_feature='age'
				lifespan_feature='lifespan'
				gfp_feature=key
			bins=worms.bin(lifespan_feature,nbins=5, equal_count=True)
			averaged_worms = worm_data.meta_worms(bins, gfp_feature, age_feature=age_feature)
			figure=averaged_worms.plot_timecourse(gfp_feature,age_feature=age_feature,min_age=min_age, max_age=max_age,time_units=time_units)
			
			cohorts=[]

			for worm, bin in bins.items():
				cohorts.append(len(bin))	
			text='n = ' + (str)(cohorts)
			ax = plt.gca() 
			ax.set_xlabel('Days', fontdict=label_fontdict, fontsize=24)
			ax.xaxis.set_label_coords(.95, .05)
			ax.set_ylabel('Expression ('+value+')',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=-.5,right=19)
			plt.figtext(.5, .15, text, fontdict=text_fontdict, fontsize=28)
			plt.title('P'+miRNA+'::GFP',fontdict=title_fontdict, loc='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
#extra
def all_wild_type_scatter_plots(root_folder,min_age=2,target='lifespan'):

	#generates single day scatter plots for multiple regression on slope and mean
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			age_feature='age'
			lifespan_feature='lifespan'
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'			
			try:			
				lifespans=worms.get_feature(lifespan_feature)
				max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				extended_max_age=(int)(round(numpy.percentile(lifespans/24,75)))
			except:
				continue	
		ages=[]
		
		for i in range(min_age,extended_max_age+1):
			ages.append(i)	
		print(ages)
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		for key,value in gfp_measures.items():
			for i in range(0,len(ages)-1):
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_scatter_plots/'+miRNA+'_'+key+'_'+(str)(ages[i+1])+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_scatter_plots/'+miRNA+'_'+key+'_'+(str)(ages[i+1])+'.svg'
				filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i+1]*24)
				lifespans=filtered_worms.get_feature('lifespan')
				try:
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
					slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(lifespans,results.y_est)
					color_vals = colorize.scale(lifespans, output_max=1)
					colors = colorize.color_map(color_vals, uint8=False)
				except:
					continue	
				try:	
					plt.scatter(lifespans/24,results.y_est/24,c=colors)
				except:
					continue		
				(m,b) = numpy.polyfit(lifespans/24, results.y_est/24, 1)
				yp=numpy.polyval([m,b], lifespans/24)
				mew, mewmew=zip(*sorted(zip(lifespans/24, yp)))
				mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
				plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
				ftext="r2 = "+(str)(round(results.R2,3))
				ptext="p = " + (str)(round(pvalue,3))
				more_text="n= " + (str)(len(lifespans))
				plt.title('Regression on Slope and Mean ' +miRNA+'::GFP expression profiles '+(str)(ages[i]) + ' to '+(str)(ages[i+1])+ ' '+key,fontdict=title_fontdict)
				plt.xlabel('Actual lifespan (days)',fontdict=label_fontdict)
				plt.ylabel("Predicted lifespan (days)",fontdict=label_fontdict)
				plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
				plt.figtext(.15,.75,ptext,fontsize=20,ha='left')
				plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
				
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()			
#Extra
def all_wild_type_correlation_plots(root_folder,overwrite=True,min_age=2, strict=True):
	
	#plots correlation coefficients over time with lifespans and ages rescaled. Gfp expression expected from z-scored data. Strict parameters require animals to be alive
	#for entire 24 hour time window. If strict is true, slope will be included in regression. Otherwise regression will only use average.
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))

				#using z scores so no scaling for gfp, just lifespan and age	
				
				for worm in worms:
						exp = worm.name.split()[0]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						age_feature='scaled_age'
						lifespan_feature='scaled_lifespan'

			else:
				for worm in worms:
					lifespan_feature='lifespan'
					age_feature='age'			
			try:			
				lifespans=worms.get_feature(lifespan_feature)

				#maximum age to calculate correlation is 90% survival day, extended max age is the 25% survival day
				
				max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				extended_max_age=(int)(round(numpy.percentile(lifespans/24,75)))
			except:
				continue	
		ages=[]
		
		for i in range(min_age,extended_max_age+1):
			ages.append(i)	
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		features_towrite.append('n')
		for key,value in gfp_measures.items():
			try:
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				correlations=[]
				p_values=[]	
				for i in range(0,len(ages)-1):
					#strict criteria: worms must survive the entire day, can use slope in regression. Not strict, worm must be alive at the beginning of the day, cannot use slope
					if strict:
						filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > ages[i+1]*24)
						results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
						correlations_towrite.append(len(filtered_worms))
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_strict/'+miRNA+'_'+key+'.svg'
						if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
							continue
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_strict/'+miRNA+'_'+key+'.svg'
					else:
						filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > ages[i]*24)
						results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
						features_towrite.append(len(filtered_worms))
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots/'+miRNA+'_'+key+'.svg'
						if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
							continue
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots/'+miRNA+'_'+key+'.svg'
								
					correlations.append(results.R2)
					slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.y_est+results.resid,results.y_est)
					p_values.append(pvalue)
				
				p_x=[]
				p_y=[]

				for i in range(0,len(p_values)):
					if p_values[i]<.05 and correlations[i]>=.1:
						p_y.append(correlations[i]+.03)
						p_x.append(ages[1::][i])	
				correlations_towrite.append(correlations)
				features_towrite.append(key)
					
				plt.scatter(p_x,p_y,marker=(6,2,0),c='navy',s=50)
				plt.scatter(ages[1::],correlations,c='navy',marker='o',s=50,edgecolor='navy')
				plt.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='navy')
				plt.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='navy',linestyle='--')
				plt.margins(.15,.15)	
				plt.title('P'+miRNA + '::GFP lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
				plt.xlabel('Age (hours post-hatch)', fontdict=label_fontdict,fontsize=24)
				plt.ylabel('Coefficient of determination (R2)',fontdict=label_fontdict)
				plt.ylim(0,.4)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()
			except:
				continue	

		
		n=len(ages)
		rows=[[] for _ in range(n)]
		for col in correlations_towrite:
			for row, colval in zip(rows,col):
					row.append(colval)
		if strict:			
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_strict/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
			datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_strict/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
		else:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
			datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots/'+miRNA+'_correlations.tsv',[features_towrite]+rows)

#Supplementary Table 3
def all_wild_type_scatter_plots_day_of_max_gfp(root_folder,min_age=3*24,target='lifespan'):
	#plots lifespan against age at peak expression. Testing 'live fast die young'.  

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'
			max_age=max_age*24
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'		
			try:			
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))
			except:
				continue	

		for key,value in gfp_measures.items():		
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_scatter_plots_day_of_max_gfp/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_scatter_plots_day_of_max_gfp/'+miRNA+'_'+key+'.svg'
			filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > max_age)
			lifespans=filtered_worms.get_feature(lifespan_feature)
			results=filtered_worms.regress(get_peak_age_wrapper(min_age,max_age,key,age_feature=age_feature),target=lifespan_feature)				
			slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X,lifespans)
			color_vals = colorize.scale(lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)
			try:	
				plt.scatter(results.X/24,lifespans/24,c=colors)
			except:
				continue		
			(m,b) = numpy.polyfit(results.X/24, lifespans/24, 1)
			yp=numpy.polyval([m,b], results.X/24)
			mew, mewmew=zip(*sorted(zip(results.X/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="r2 = "+(str)(results.R2)
			ptext="p = " + (str)(pvalue)
			more_text="n= " + (str)(len(lifespans))
			plt.title('Peak ' +miRNA+'::GFP expression v. lifespan',fontdict=title_fontdict)
			plt.xlabel('Age of Peak expression ('+key+')',fontdict=label_fontdict, fontsize=24)
			plt.ylabel("Lifespan (days)",fontdict=label_fontdict, fontsize=24)
			plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			plt.figtext(.15,.75,ptext,fontsize=20,ha='left')
			plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()

#Supplementary Table 3
def all_wild_type_scatter_plots_max_gfp(root_folder,min_age=3*24):
	#plots lifespan against maximum value of expression

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
				for exp, wormies in groups.items():
					data = wormies.get_time_range(key,min_age,numpy.inf)
					median_gfp[exp] = numpy.median([numpy.median(d[:,1]) for d in data])
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
					exp = worm.name.split()[0]
					worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
					worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
					worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
					gfp_feature='scaled_gfp'
					lifespan_feature='scaled_lifespan'
					age_feature='scaled_age'
			else:
				gfp_feature=key
				lifespan_feature='lifespan'
				age_feature='age'

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24
			else:				
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))
				
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_scatter_plots_max_gfp/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_scatter_plots_max_gfp/'+miRNA+'_'+key+'.svg'
			filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > max_age)
			lifespans=filtered_worms.get_feature(lifespan_feature)
			results=filtered_worms.regress(get_peak_wrapper(min_age,max_age,gfp_feature,age_feature=age_feature),target=lifespan_feature)			
			slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X, lifespans)
			color_vals = colorize.scale(lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)
			try:	
				plt.scatter(results.X,lifespans/24,c=colors)
			except:
				continue		
			(m,b) = numpy.polyfit(results.X, lifespans/24, 1)
			yp=numpy.polyval([m,b], results.X)
			mew, mewmew=zip(*sorted(zip(results.X, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="r2 = "+(str)(results.R2)
			ptext="p = " + (str)(pvalue)
			more_text="n= " + (str)(len(lifespans))
			plt.title('Peak ' +miRNA+'::GFP expression v. lifespan',fontdict=title_fontdict)
			plt.xlabel('Peak expression ('+key+')',fontdict=label_fontdict, fontsize=24)
			plt.ylabel("Lifespan (days)",fontdict=label_fontdict, fontsize=24)
			plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			plt.figtext(.15,.75,ptext,fontsize=20,ha='left')
			plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()			
#Figure 1B
def all_wild_type_plot_individual_trajectory(root_folder,min_age=3*24):
	#picks a random individual from each experiment directory and plots its expression trajectory over time. Useful for seeing
	#how much an individual resembles population level trace 

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		for exp_dir in exp_dirs:
			worms=process_worms([exp_dir], prefixes=[''])
			random_worm=worms[numpy.random.random_integers(0,len(worms)-1)]
			for key,value in gfp_measures.items():
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_individual_trajectory/'+miRNA+'_'+exp_dir[-9::]+'_'+key+'_'+(str)(random_worm)+'.svg'
				ages,data=random_worm.get_time_range(key,min_age=24)
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_individual_trajectory/'+miRNA+'_'+exp_dir[-9::]+'_'+key+'_'+(str)(random_worm)+'.svg'
				plt.plot(ages/24,data)
				plt.scatter(ages/24,data)
				plt.title('P' +miRNA+'::GFP expression for '+(str)(random_worm),fontdict=title_fontdict)
				plt.xlabel('Age (days)',fontdict=label_fontdict, fontsize=24)
				plt.ylabel('Expression ('+value+')',fontdict=label_fontdict, fontsize=24)
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()

#Figure 3, Supplementary Figure 3				
def all_wild_type_correlation_plots_with_survival_curve(subdirectories,min_age=2, strict=True,rescale_lifespan=True,filter_near_dead=False):
	
	#plots correlation coefficients over time with survival curve. Default uses strict criteria (worms must be alive entire day) and rescales age and lifespan.
	#filter_near_dead parameter will provide additional correlation plots filtering out animals who die in the next 48 and 72 hours
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))

				#using z scores so no scaling for gfp, just lifespan and age	
				
				for worm in worms:
					exp = worm.name.split()[0]
					if rescale_lifespan:
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						age_feature='scaled_age'
						lifespan_feature='scaled_lifespan'
					else:	
						age_feature='age'
						lifespan_feature='lifespan'

			else:
				lifespan_feature='lifespan'
				age_feature='age'			
			try:			
				lifespans=worms.get_feature(lifespan_feature)

				#maximum age to calculate correlation is 90% survival day, extended max age is the 25% survival day
				
				max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				extended_max_age=(int)(round(numpy.percentile(lifespans/24,75)))
			except:
				continue	
		ages=[]
		
		for i in range(min_age,extended_max_age+1):
			ages.append(i)	
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			correlations=[]
			correlations_1=[]
			correlations_2=[]
			p_values=[]
			p_values_1=[]
			p_values_2=[]
			#try:		
			for i in range(0,len(ages)-1):
				#strict criteria: worms must survive the entire day, can use slope in regression. Not strict, worm must be alive at the beginning of the day, cannot use slope
				if strict:
					filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > ages[i+1]*24)
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
					if rescale_lifespan:
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
					else:
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_strict/'+miRNA+'_'+key+'.svg'	
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_strict/'+miRNA+'_'+key+'.svg'
				if filter_near_dead:
					filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > (ages[i+1])*24)
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
					filtered_worms_1=worms.filter(lambda worm: getattr(worm,lifespan_feature) > (ages[i+1]+1)*24)
					filtered_worms_2=worms.filter(lambda worm: getattr(worm,lifespan_feature) > (ages[i+1]+2)*24)
					if len(filtered_worms_1)>25:
						results_1=filtered_worms_1.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
						slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results_1.y_est+results_1.resid,results_1.y_est)
						correlations_1.append(results_1.R2)
						p_values_1.append(pvalue)
					if len(filtered_worms_2)>25:
						results_2=filtered_worms_2.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
						correlations_2.append(results_2.R2)
						slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results_2.y_est+results_2.resid,results_2.y_est)
						p_values_2.append(pvalue)					

					if rescale_lifespan:
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
					else:
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead/'+miRNA+'_'+key+'.svg'	
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead/'+miRNA+'_'+key+'.svg'
	
				if strict==False and filter_near_dead==False:
					filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > ages[i]*24)
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
					if rescale_lifespan:
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
					else:	
						save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve/'+miRNA+'_'+key+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve/'+miRNA+'_'+key+'.svg'
							
				correlations.append(results.R2)
				slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.y_est+results.resid,results.y_est)
				p_values.append(pvalue)
			
			p_x=[]
			p_y=[]

			for i in range(0,len(p_values)):
				if p_values[i]<.05 and correlations[i]>=.1:
					p_y.append(correlations[i]+.03)
					p_x.append(ages[1::][i])
			try:
				p_x_1=[]
				p_y_1=[]
				p_x_2=[]
				p_y_2=[]		
				for i in range(0,len(p_values_1)):
					if p_values_1[i]<.05 and correlations_1[i]>=.1:
						p_y_1.append(correlations_1[i]+.03)
						p_x_1.append(ages[1::][i])
					if p_values_2[i]<.05 and correlations_2[i]>=.1:
						p_y_2.append(correlations_2[i]+.03)
						p_x_2.append(ages[1::][i])
			except:
				print('')								
			correlations_towrite.append(correlations)
			features_towrite.append(key)

			lifespans=worms.get_feature(lifespan_feature)/24
			max_life=lifespans.max()
			min_life=lifespans.min()
			days=numpy.arange(3,max_life+1,.25)
			percent_alive = []
			for i in days:
				count =0
				for item in lifespans:
					if item > i:
						count=count+1
				percent_alive.append((count/len(lifespans))*100)
			plt.rcParams['axes.spines.right']=True	
			fig, ax1= plt.subplots()
			color = 'gray'
			ax1.set_xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict, fontsize=24)
			ax1.set_ylabel('Survival (%)', fontdict=label_fontdict, color=color, fontsize=24)
			ax1.tick_params(axis='y', labelcolor=color)	
			ax1.plot(days,percent_alive,color=color,alpha=.5)
			ax2 = ax1.twinx()
			color = 'green'
			ax2.set_ylabel('Coefficient of determination (R2)',fontdict=label_fontdict, color=color, fontsize=24)
			ax2.scatter(p_x,p_y,marker=(6,2,0),c='green',s=50)
			ax2.scatter(ages[1::],correlations,c='green',marker='o',s=25,edgecolor='green')
			try:
				ax2.scatter(ages[1:(len(correlations_1)+1)],correlations_1,c='blue',marker='o',s=25,edgecolor='blue')
				ax2.plot(ages[1:ages.index(max_age)+1],correlations_1[0:ages.index(max_age)], c='blue')
				ax2.plot(ages[ages.index(max_age):(len(correlations_1)+1)],correlations_1[ages.index(max_age)-1:(len(correlations_1)+1)],c='blue',alpha=.5)
				ax2.scatter(p_x_1,p_y_1,marker=(6,2,0),c='blue',s=50)
				ax2.scatter(ages[1:(len(correlations_2)+1)],correlations_2,c='red',marker='o',s=25,edgecolor='red')
				ax2.plot(ages[1:ages.index(max_age)+1],correlations_2[0:ages.index(max_age)], c='red')
				ax2.plot(ages[ages.index(max_age):(len(correlations_2)+1)],correlations_2[ages.index(max_age)-1:(len(correlations_2)+1)],c='red',alpha=.5)
				ax2.scatter(p_x_2,p_y_2,marker=(6,2,0),c='red',s=50)
			except:
				print('')
			ax2.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='green')
			ax2.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='green',alpha=.5)
			ax2.tick_params(axis='y', labelcolor=color)	
			ax2.set_ylim(0,.4)
			plt.title('P'+miRNA + '::GFP lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
			#except:
				#continue
		
		n=len(ages)
		rows=[[] for _ in range(n)]
		for col in correlations_towrite:
			for row, colval in zip(rows,col):
					row.append(colval)
		if strict:
			if rescale_lifespan:			
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
			else:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_strict/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_strict/'+miRNA+'_correlations.tsv',[features_towrite]+rows)	
		if filter_near_dead:
			if rescale_lifespan: 
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)	
			else:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_filter_near_dead/'+miRNA+'_correlations.tsv',[features_towrite]+rows)	
		
		if strict==False and filter_near_dead==False:
			if rescale_lifespan:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)	
			else:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve/'+miRNA+'_correlations.tsv',[features_towrite]+rows)

#Supplementary Figure 5
def all_wild_type_correlation_plots_with_auto_with_survival_curve(root_folder,min_age=2,target='lifespan',strict=True, rescale_lifespan=True):

	#plots correlation coefficients over time for gfp and autofluorescence with survival curve. Default uses strict criteria (worms must be alive entire day) and rescales age and lifespan
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'	
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				#using z scores so no scaling for gfp, just lifespan and age
				for worm in worms:
					exp = worm.name.split()[0]
					if rescale_lifespan:
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
					else:	
						age_feature='age'
						lifespan_feature='lifespan'
			else:
				lifespan_feature='lifespan'
				age_feature='age'					
			try:			
				lifespans=worms.get_feature(lifespan_feature)
				#maximum age to calculate correlation is 90% survival day, extended max age is the 25% survival day
				max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				extended_max_age=(int)(round(numpy.percentile(lifespans/24,75)))
			except:
				continue	
		ages=[]
		
		for i in range(min_age,extended_max_age+1):
			ages.append(i)	
		print(ages)
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			correlations=[]
			auto_correlations=[]
			p_values=[]
			auto_p_values=[]
			try:
				for i in range(0,len(ages)-1):
				#strict criteria: worms must survive the entire day, can use slope in regression. Not strict, worm must be alive at the beginning of the day, cannot use slope
					if strict:
						filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > ages[i+1]*24)
						results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
						results_auto=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,'green_yellow_excitation_autofluorescence_percentile_95_z',age_feature=age_feature),target=lifespan_feature)
						if rescale_lifespan:
							save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
							save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
						else:
							save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict/'+miRNA+'_'+key+'.svg'	
							save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict/'+miRNA+'_'+key+'.svg'
					else:
						filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > ages[i]*24)
						results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
						results_auto=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,'green_yellow_excitation_autofluorescence_percentile_95_z',age_feature=age_feature),target=lifespan_feature)
						if rescale_lifespan:
							save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
							save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_auto_with_survival_curve_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
						else:	
							save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve/'+miRNA+'_'+key+'.svg'
							save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_auto_with_survival_curve/'+miRNA+'_'+key+'.svg'								
					correlations.append(results.R2)
					auto_correlations.append(results_auto.R2)
					slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.y_est+results.resid,results.y_est)
					p_values.append(pvalue)
					slope, intercept,rvalue,pvalue_auto,stderror=scipy.stats.linregress(results_auto.y_est+results_auto.resid,results_auto.y_est)
					auto_p_values.append(pvalue_auto)
			except:
				continue		

			correlations_towrite.append(correlations)
			features_towrite.append(key)
			p_x=[]
			p_y=[]
			p_x_auto=[]
			p_y_auto=[]

			for i in range(0,len(p_values)):
				if p_values[i]<.05 and correlations[i]>=.1:
					p_y.append(correlations[i]+.02)
					p_x.append(ages[1::][i])
				if auto_p_values[i]<.05 and auto_correlations[i]>=.1:
					p_x_auto.append(ages[1::][i])
					p_y_auto.append(auto_correlations[i]+.02)
	
			lifespans=worms.get_feature(lifespan_feature)/24
			max_life=lifespans.max()
			min_life=lifespans.min()
			days=numpy.arange(3,max_life+1,.25)
			percent_alive = []	
			for i in days:
				count =0
				for item in lifespans:
					if item > i:
						count=count+1
				percent_alive.append((count/len(lifespans))*100)
			plt.rcParams['axes.spines.right']=True	
			fig, ax1= plt.subplots()
			color = 'gray'
			ax1.set_xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict, fontsize=24)
			ax1.set_ylabel('Survival (%)', fontdict=label_fontdict, color=color, fontsize=24)
			ax1.tick_params(axis='y', labelcolor=color)	
			ax1.plot(days,percent_alive,color=color,alpha=.5)
			ax2 = ax1.twinx()
			color = 'green'
			ax2.set_ylabel('Coefficient of determination (R2)',fontdict=label_fontdict, color=color, fontsize=24)
			ax2.scatter(p_x,p_y,marker=(6,2,0),c='green',s=25)
			ax2.scatter(ages[1::],correlations,c='green',marker='o',s=50,edgecolor='green')
			ax2.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='green')
			ax2.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='green',alpha=.5)
			ax2.tick_params(axis='y', labelcolor=color)	
			ax2.set_ylim(0,.4)
			ax2.scatter(p_x_auto,p_y_auto,marker=(6,2,0),c='red',s=25)
			ax2.scatter(ages[1::],auto_correlations,c='red',marker='o',s=50,edgecolor='red')
			ax2.plot(ages[1:ages.index(max_age)+1],auto_correlations[0:ages.index(max_age)], c='red')
			ax2.plot(ages[ages.index(max_age)::],auto_correlations[ages.index(max_age)-1::],c='red', alpha=.5)
			red_patch = mpatches.Patch(color='red', label='autofluorescence (95th percentile)')
			green_patch = mpatches.Patch(color='green',label='GFP'+' '+key)
			plt.legend(handles=[red_patch,green_patch])		
			plt.title('P'+miRNA + '::GFP lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)	
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
		n=len(ages)
		rows=[[] for _ in range(n)]
		for col in correlations_towrite:
			for row, colval in zip(rows,col):
				row.append(colval)
		if strict:
			if rescale_lifespan:			
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
			else:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_strict/'+miRNA+'_correlations.tsv',[features_towrite]+rows)	
		else:
			if rescale_lifespan:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve_lifespan_rescaled/'+miRNA+'_correlations.tsv',[features_towrite]+rows)	
			else:
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve/'+miRNA+'_correlations.tsv',[features_towrite]+rows)
				datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve/'+miRNA+'_correlations.tsv',[features_towrite]+rows)

#Supplementary Table 1
def all_wild_type_regression_on_slope_and_mean_same_x(root_folder,min_age=3*24,target='lifespan', rescale_lifespan=True,bootstrap=False):

	#performs ordinary least squares regression on slope and mean of expression in a defined time window.
	#lifespan and age are rescaled to median. Z scores are used for gfp measurements
	#everything is plotted using same x axis

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			if rescale_lifespan:
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_same_x_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_same_x_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
			else:
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'	
			
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						if rescale_lifespan:
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							lifespan_feature='scaled_lifespan'
							age_feature='scaled_age'
						else:
							lifespan_feature='lifespan'
							age_feature='age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'												

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24
			else:
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))
				

			#right endpoint of time window calculated based on 90% survival				

			lifespans=worms.get_feature(lifespan_feature)		
			#only worms that are alive in the entire window are included in the regression
			#try:
			filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature)> max_age)
			filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
			results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			correlations_to_write.append(slope_results.R2)
			correlations_to_write.append(average_results.R2)
			correlations_to_write.append(results.R2)
			features_to_write.append(key+'_slope')
			features_to_write.append(key+'_mean')
			features_to_write.append(key+'_joint')
			if bootstrap==True:
				bootstrapped_correlations=[]
				bootstrapped_slope_correlations=[]
				bootstrapped_average_correlations=[]
				for i in range(0,1000):
					sample_worms=numpy.random.choice(filtered_worms,len(filtered_worms),replace=True)
					sample_worms=worm_data.Worms(sample_worms)	
					bootstrapped_results=sample_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_correlations.append(bootstrapped_results.R2)
					bootstrapped_slope_results=sample_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_slope_correlations.append(bootstrapped_slope_results.R2)
					bootstrapped_average_results=sample_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_average_correlations.append(bootstrapped_average_results.R2)	
			
				correlations_to_write.append([2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,97.5), 2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,97.5), 2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,97.5), 2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,2.5)])
				features_to_write.append(key+'_bootstrapped_slope')
				features_to_write.append(key+'_bootstrapped_mean')
				features_to_write.append(key+'_bootstrapped_joint')
			#except:	
				#continue
			#plot true lifespans vs. lifespans estimated by multiple regression
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(results.y_est+results.resid,results.y_est)		
			color_vals = colorize.scale(results.y_est+results.resid, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter((results.y_est+results.resid)/24,results.y_est/24,c=colors)
			(m,b) = numpy.polyfit((results.y_est+results.resid)/24, results.y_est/24, 1)
			yp=numpy.polyval([m,b], (results.y_est+results.resid)/24)
			mew, mewmew=zip(*sorted(zip((results.y_est+results.resid)/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="R2 = "+(str)(round(results.R2,3))
		
			#features_to_write.append(key+'_p_value')
			#correlations_to_write.append(p_value)
			ptext="p = " + (str)(round(p_value, 3))
			more_text="n= " + (str)(len((results.y_est+results.resid)))
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			#plot everything on the same x scale
			ax.set_xlim(left=100/24, right=450/24)
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.savefig
			plt.gcf().clf
			plt.close()
		#write out correlation coefficients and p values to tsv file	
		features_to_write.append('n')
		features_to_write.append('max_age')	
		correlations_to_write.append(len(results.y_est+results.resid))
		correlations_to_write.append(max_age)	
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		if rescale_lifespan:		
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_same_x_lifespan_rescaled/'+miRNA+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])
		else:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_same_x/'+miRNA+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])	

#Supplementary Table 6
def all_wild_type_regression_on_slope_and_mean_individual_experiments(root_folder,min_age=3*24,target='lifespan',bootstrap=False):
	#performs ordinary least squares regression on individual experiment directories
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])

		
		for exp_dir in exp_dirs:
			features_to_write=[]
			correlations_to_write=[]
			for key,value in gfp_measures.items():
				worms=process_worms([exp_dir], prefixes=[' '])
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_individual_experiments/'+miRNA+'_'+exp_dir[-9::]+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_individual_experiments/'+miRNA+'_'+exp_dir[-9::]+'_'+key+'.svg'
				
				if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
					header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
					parameters=list(parameters)
					max_age=parameters[header.index('max_age')]
					max_age=((int)(max_age[0]))*24			
				
				else:
					lifespans=worms.get_feature(target)
					max_age=round(numpy.percentile(lifespans,10))	
			
				lifespans=worms.get_feature(target)
			
				try:
					filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
					filtered_lifespans=filtered_worms.get_feature('lifespan')
	
					results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key),get_slope_wrapper(min_age,max_age,key),target=target)
					slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key),target=target)
					average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key),target=target)
					correlations_to_write.append(slope_results.R2)
					correlations_to_write.append(average_results.R2)
					correlations_to_write.append(results.R2)
					features_to_write.append(key+'_slope')
					features_to_write.append(key+'_mean')
					features_to_write.append(key+'_joint')
				except:	
					continue

				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)
				features_to_write.append('p value')
				correlations_to_write.append(p_value)		
				color_vals = colorize.scale(filtered_lifespans, output_max=1)
				colors = colorize.color_map(color_vals, uint8=False)	
				plt.scatter(filtered_lifespans/24,results.y_est/24,c=colors)
				(m,b) = numpy.polyfit(filtered_lifespans/24, results.y_est/24, 1)
				yp=numpy.polyval([m,b], filtered_lifespans/24)
				mew, mewmew=zip(*sorted(zip(filtered_lifespans/24, yp)))
				mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
				plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
				ftext="R2 = "+(str)(round(results.R2,3))
				ptext="p = " + (str)(round(p_value, 3))
				more_text="n= " + (str)(len(filtered_lifespans))
				plt.title(miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
				ax = plt.gca() 
				ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict)
				ax.xaxis.set_label_coords(.85, .05)
				ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict)
				ax.yaxis.set_label_coords(.05,.5)
				plt.figtext(.15, .15, more_text, fontsize=24, ha='left')
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.savefig
				plt.gcf().clf
				plt.close()
			features_to_write.append('n')
			features_to_write.append('max_age')	
			correlations_to_write.append(len(filtered_lifespans))
			correlations_to_write.append(max_age)	
			rows=[[] for _ in range(len(features_to_write))]
			for col in features_to_write:	
				for row, colval in zip(rows,correlations_to_write):
					row.append(colval)
	
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_individual_experiments/'+miRNA+'_'+exp_dir[-9::]+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])			

#Figure 2B
def all_wild_type_regression_on_slope_and_mean_sliding(root_folder,min_age=3*24,target='lifespan',rescale_lifespan=True,bootstrap=False):
	#performs ordinary least squares regression using a sliding 12 hour time window. Default is to rescale lifespan and age. Z scores are used for gfp measurements
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			if rescale_lifespan:
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_sliding_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_sliding_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
			else:
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						#rescale lifespan and age if rescale_lifespan=True
						if rescale_lifespan:
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							lifespan_feature='scaled_lifespan'
							age_feature='scaled_age'
						else:
							lifespan_feature='lifespan'
							age_feature='age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'						

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))	
		
			ages=[]
			correlations=[]
			windows=[]
			#create list of timepoints 12 hours apart within min_age and 90% survival timepoint
			for i in range(min_age,(int)(max_age)+(int)(max_age)%12,12):
				ages.append(i)	
			for i in range(0, len(ages)-1):
				left_window=ages[i]
				for j in range(ages.index(left_window)+1, len(ages)):
					right_window=ages[j]
					#worms have to be alive in the specified window
					filtered_worms=worms.filter(lambda worm: getattr(worm, lifespan_feature) > right_window)
					filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
					try:
						results=filtered_worms.regress(get_average_wrapper(left_window,right_window,key+'_z',age_feature=age_feature),get_slope_wrapper(left_window,right_window,key+'_z',age_feature=age_feature),target=lifespan_feature)
					except:
						continue
					correlations.append(results.R2)
					windows.append([left_window,right_window])
			#pick time window where maximum r2 value is achieved	
			best_window=windows[correlations.index(max(correlations))]
			results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),get_slope_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)		
			slope_results=filtered_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
			average_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
			correlations_to_write.append(slope_results.R2)
			correlations_to_write.append(average_results.R2)
			correlations_to_write.append(results.R2)
			features_to_write.append(key+'_slope')
			features_to_write.append(key+'_mean')
			features_to_write.append(key+'_joint')
			if bootstrap==True:
				bootstrapped_correlations=[]
				bootstrapped_slope_correlations=[]
				bootstrapped_average_correlations=[]
				for i in range(0,1000):
					sample_worms=numpy.random.choice(filtered_worms,len(filtered_worms),replace=True)
					sample_worms=worm_data.Worms(sample_worms)	
					bootstrapped_results=sample_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),get_slope_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_correlations.append(bootstrapped_results.R2)
					bootstrapped_slope_results=sample_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_slope_correlations.append(bootstrapped_slope_results.R2)
					bootstrapped_average_results=sample_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_average_correlations.append(bootstrapped_average_results.R2)	
			
				correlations_to_write.append([2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,97.5), 2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,97.5), 2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,97.5), 2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,2.5)])
				features_to_write.append(key+'_bootstrapped_slope')
				features_to_write.append(key+'_bootstrapped_mean')
				features_to_write.append(key+'_bootstrapped_joint')
			#features_to_write.append(key+'_joint_p_value')
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(results.y_est+results.resid,results.y_est)	
			#correlations_to_write.append(p_value)
			correlations_to_write.append(len(results.resid))
			features_to_write.append('n')
			features_to_write.append('best_window')
			correlations_to_write.append(best_window)	
			color_vals = colorize.scale((results.y_est+results.resid), output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter((results.y_est+results.resid)/24,results.y_est/24,c=colors)
			(m,b) = numpy.polyfit((results.y_est+results.resid)/24, results.y_est/24, 1)
			yp=numpy.polyval([m,b], (results.y_est+results.resid)/24)
			mew, mewmew=zip(*sorted(zip((results.y_est+results.resid)/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="R2 = "+(str)(round(results.R2,3))
			more_text="n= " + (str)(len(results.resid))
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=100/24, right=450/24)
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.savefig
			plt.gcf().clf
			plt.close()
	
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		if rescale_lifespan:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_sliding_lifespan_rescaled/'+miRNA+'_'+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])		
		else:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_sliding/'+miRNA+'_'+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])

#Supplementary Table 4
def all_wild_type_regression_on_slope_and_mean_plus_autofluorescence(root_folder,min_age=3*24,target='lifespan'):
	#Adds mean and slope of 95th percentile intensity autofluorescence to regression
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence/'+miRNA+'_'+key+'.svg'
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'			

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24

			else:
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(lifespan_feature)
			
		
			filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > max_age)
			gfp_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			auto_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z',age_feature=age_feature),target=lifespan_feature)
			gfp_plus_auto_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z',age_feature=age_feature),target=lifespan_feature)
			filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
			features_to_write.append(key+'_gfp')
			correlations_to_write.append(gfp_results.R2)
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,gfp_results.y_est)
			features_to_write.append(key+'_gfp_p_value')
			correlations_to_write.append(p_value)
			features_to_write.append('auto')
			correlations_to_write.append(auto_results.R2)
			features_to_write.append(key+'_auto_p_value')
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,auto_results.y_est)
			correlations_to_write.append(p_value)
			features_to_write.append(key+'_gfp_plus_auto')
			correlations_to_write.append(gfp_plus_auto_results.R2)
			features_to_write.append(key+'_gfp_plus_auto_p_value')
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,gfp_plus_auto_results.y_est)
			correlations_to_write.append(p_value)
			filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,gfp_plus_auto_results.y_est)	
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter(filtered_lifespans/24,gfp_plus_auto_results.y_est/24,c=colors)
			(m,b) = numpy.polyfit(filtered_lifespans/24, gfp_plus_auto_results.y_est/24, 1)
			yp=numpy.polyval([m,b], filtered_lifespans/24)
			mew, mewmew=zip(*sorted(zip(filtered_lifespans/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			more_text="n= " + (str)(len(filtered_lifespans))
			plt.title('P'+miRNA+'::GFP expression and autofluorescence vs. lifespan: R2='+(str)(round(gfp_plus_auto_results.R2,3)),fontdict=title_fontdict)
			plt.xlabel('Lifespan (days)',fontdict=label_fontdict,fontsize=24)
			plt.ylabel("Expression-predicted lifespan (days)",fontdict=label_fontdict,fontsize=24)
			plt.figtext(.8, .2, more_text, fontsize=24, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
		features_to_write.append('n')
		features_to_write.append('max_age')
		filtered_lifespans=filtered_worms.get_feature(lifespan_feature)	
		correlations_to_write.append(len(filtered_lifespans))
		correlations_to_write.append(max_age)	
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence/'+miRNA+ '_regression_correlations.tsv',[features_to_write]+[correlations_to_write])

#Supplementary Figure 4
def all_wild_type_population_trajectory_with_stddev(root_folder,min_age=24,time_units='hours'):
	#plots population level traces with standard deviation
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
		else:
			max_age=numpy.inf		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory_with_stddev/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory_with_stddev/'+miRNA+'_'+key+'.svg'
			try:
				worms.get_time_range(key,24,numpy.inf)
			except:
				continue	
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,max_age)	
						median_lifespans[exp]=numpy.median(wormies.get_feature('lifespan'))
						median_gfp[exp] = numpy.median([d[:,1].mean() for d in data])
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						age_feature='scaled_age'
						lifespan_feature='scaled_lifespan'
						gfp_feature='scaled_gfp'

			else:
				gfp_feature=key
				lifespan_feature='lifespan'
				age_feature='age'
			lifespans=worms.get_feature(lifespan_feature)
			bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}
			averaged_worms = worm_data.meta_worms(bins, gfp_feature, age_feature=age_feature)	
			cohorts=[]
			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			trend_x,mean_trend,std_trend=worms.z_transform(gfp_feature,min_age=72,max_age=max_age,age_feature=age_feature)
			plt.plot(trend_x/24,mean_trend,c='mediumspringgreen')
			plt.plot(trend_x/24,mean_trend+std_trend,c='mediumspringgreen',alpha=.5,linestyle='--')
			plt.plot(trend_x/24,mean_trend-std_trend,c='mediumspringgreen',alpha=.5,linestyle='--')
			text='n = ' + (str)(cohorts)
			plt.figtext(.5, .15, text, fontdict=text_fontdict,fontsize=28)
			plt.title('P' + miRNA+ '::GFP expression',fontdict=title_fontdict)
			plt.ylabel('Expression ('+value+')',fontdict=label_fontdict,fontsize=24)
			plt.xlabel('Days',fontdict=label_fontdict,fontsize=24)
			ax = plt.gca() 
			ax.set_xlim(left=-.5,right=19)
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
								
"""non miRNA functions"""

#Supplemental Figure 2

def all_nonmir_population_trajectory_cohorts(root_folder,overwrite=True,min_age=24,time_units='days',rescale_lifespan=True):
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		max_age=numpy.inf	
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
			lifespan_feature='lifespan'
			age_feature='age'	
		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			if rescale_lifespan:
				save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_population_trajectory_cohorts_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_population_trajectory_cohorts_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
			else:
				save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'	
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			try:
				worms.get_time_range(key,min_age,numpy.inf)
			except:
				continue	
			
			#rescales each experiment to median gfp, also lifespan and age if there's more than one experiment directory to facilitate pooling
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,numpy.inf)	
						median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
						median_gfp[exp] = numpy.median([numpy.median(d[:,1]) for d in data])
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						if rescale_lifespan:
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							age_feature='scaled_age'
							lifespan_feature='scaled_lifespan'
							gfp_feature='scaled_gfp'
						else:
							age_feature='age'
							lifespan_feature='lifespan'
							gfp_feature=key
			else:
				age_feature='age'
				lifespan_feature='lifespan'
				gfp_feature=key
			bins=worms.bin(lifespan_feature,nbins=5, equal_count=True)
			averaged_worms = worm_data.meta_worms(bins, gfp_feature, age_feature=age_feature)
			figure=averaged_worms.plot_timecourse(gfp_feature,age_feature=age_feature,min_age=min_age, max_age=max_age,time_units=time_units)
			
			cohorts=[]

			for worm, bin in bins.items():
				cohorts.append(len(bin))	
			text='n = ' + (str)(cohorts)
			ax = plt.gca() 
			ax.set_xlabel('Days', fontdict=label_fontdict, fontsize=24)
			ax.xaxis.set_label_coords(.95, .05)
			ax.set_ylabel('Expression ('+value+')',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=-.5,right=19)
			plt.figtext(.5, .15, text, fontdict=text_fontdict, fontsize=28)
			plt.title('P'+miRNA+'::GFP',fontdict=title_fontdict, loc='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
#Supplemental Table 2
def all_nonmir_regression_on_slope_and_mean_same_x(root_folder,min_age=3*24,target='lifespan', rescale_lifespan=True,bootstrap=False):

	#performs ordinary least squares regression on slope and mean of expression in a defined time window.
	#lifespan and age are rescaled to median. Z scores are used for gfp measurements

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			if rescale_lifespan:
				save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_same_x_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_regression_on_slope_and_mean_same_x_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
			else:
				save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'	
			
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						if rescale_lifespan:
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							lifespan_feature='scaled_lifespan'
							age_feature='scaled_age'
						else:
							lifespan_feature='lifespan'
							age_feature='age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'												

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24
			else:
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))
				

			#right endpoint of time window calculated based on 90% survival				

			lifespans=worms.get_feature(lifespan_feature)		
			#only worms that are alive in the entire window are included in the regression
			try:
				filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature)> max_age)
				filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
				slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
				average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
			except:	
				continue
			if bootstrap==True:
				bootstrapped_correlations=[]
				bootstrapped_slope_correlations=[]
				bootstrapped_average_correlations=[]
				for i in range(0,1000):
					sample_worms=numpy.random.choice(filtered_worms,len(filtered_worms),replace=True)
					sample_worms=worm_data.Worms(sample_worms)	
					bootstrapped_results=sample_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_correlations.append(bootstrapped_results.R2)
					bootstrapped_slope_results=sample_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_slope_correlations.append(bootstrapped_slope_results.R2)
					bootstrapped_average_results=sample_worms.regress(get_average_wrapper(min_age,max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_average_correlations.append(bootstrapped_average_results.R2)	
			
				correlations_to_write.append([2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,97.5), 2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,97.5), 2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,97.5), 2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,2.5)])
				features_to_write.append(key+'_bootstrapped_slope')
				features_to_write.append(key+'_bootstrapped_mean')
				features_to_write.append(key+'_bootstrapped_joint')	
			#plot true lifespans vs. lifespans estimated by multiple regression
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(results.y_est+results.resid,results.y_est)		
			color_vals = colorize.scale(results.y_est+results.resid, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter((results.y_est+results.resid)/24,results.y_est/24,c=colors)
			(m,b) = numpy.polyfit((results.y_est+results.resid)/24, results.y_est/24, 1)
			yp=numpy.polyval([m,b], (results.y_est+results.resid)/24)
			mew, mewmew=zip(*sorted(zip((results.y_est+results.resid)/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="R2 = "+(str)(round(results.R2,3))
			#calculate p_value using F test
			features_to_write.append(key+'_p_value')
			correlations_to_write.append(p_value)
			ptext="p = " + (str)(round(p_value, 3))
			more_text="n= " + (str)(len((results.y_est+results.resid)))
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			#plot everything on the same x scale
			ax.set_xlim(left=100/24, right=450/24)
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.savefig
			plt.gcf().clf
			plt.close()
		#write out correlation coefficients and p values to tsv file	
		features_to_write.append('n')
		features_to_write.append('max_age')	
		correlations_to_write.append(len(results.y_est+results.resid))
		correlations_to_write.append(max_age)	
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		if rescale_lifespan:		
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_same_x_lifespan_rescaled/'+miRNA+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])
		else:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_same_x/'+miRNA+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])			

#Supplemental Table 2, Supplemental Figure 2

def all_nonmir_regression_on_slope_and_mean_sliding(root_folder,min_age=3*24,target='lifespan',rescale_lifespan=True):
	#performs ordinary least squares regression using a sliding time window
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			if rescale_lifespan:
				save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_sliding_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_regression_on_slope_and_mean_sliding_lifespan_rescaled/'+miRNA+'_'+key+'.svg'
			else:
				save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						#rescale lifespan and age if rescale_lifespan=True
						if rescale_lifespan:
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							lifespan_feature='scaled_lifespan'
							age_feature='scaled_age'
						else:
							lifespan_feature='lifespan'
							age_feature='age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'						

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(lifespan_feature)
				max_age=round(numpy.percentile(lifespans,10))	
		
			ages=[]
			correlations=[]
			windows=[]
			#create list of timepoints 12 hours apart within min_age and 90% survival timepoint
			for i in range(min_age,(int)(max_age)+(int)(max_age)%12,12):
				ages.append(i)	
			for i in range(0, len(ages)-1):
				left_window=ages[i]
				for j in range(ages.index(left_window)+1, len(ages)):
					right_window=ages[j]
					#worms have to be alive in the specified window
					filtered_worms=worms.filter(lambda worm: getattr(worm, lifespan_feature) > right_window)
					filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
					try:
						results=filtered_worms.regress(get_average_wrapper(left_window,right_window,key+'_z',age_feature=age_feature),get_slope_wrapper(left_window,right_window,key+'_z',age_feature=age_feature),target=lifespan_feature)
					except:
						continue
					correlations.append(results.R2)
					windows.append([left_window,right_window])
			#pick time window where maximum r2 value is achieved	
			best_window=windows[correlations.index(max(correlations))]
			results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),get_slope_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)		
			slope_results=filtered_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
			average_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z',age_feature=age_feature),target=lifespan_feature)
			correlations_to_write.append(slope_results.R2)
			correlations_to_write.append(average_results.R2)
			correlations_to_write.append(results.R2)
			features_to_write.append(key+'_slope')
			features_to_write.append(key+'_mean')
			features_to_write.append(key+'_joint')
			features_to_write.append(key+'_joint_p_value')
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(results.y_est+results.resid,results.y_est)	
			correlations_to_write.append(p_value)
			correlations_to_write.append(len(results.resid))
			features_to_write.append('n')
			features_to_write.append('best_window')
			correlations_to_write.append(best_window)	
			color_vals = colorize.scale((results.y_est+results.resid), output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter((results.y_est+results.resid)/24,results.y_est/24,c=colors)
			(m,b) = numpy.polyfit((results.y_est+results.resid)/24, results.y_est/24, 1)
			yp=numpy.polyval([m,b], (results.y_est+results.resid)/24)
			mew, mewmew=zip(*sorted(zip((results.y_est+results.resid)/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="R2 = "+(str)(round(results.R2,3))
			more_text="n= " + (str)(len(results.resid))
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=100/24, right=450/24)
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.savefig
			plt.gcf().clf
			plt.close()
	
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		if rescale_lifespan:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_sliding_lifespan_rescaled/'+miRNA+'_'+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])		
		else:
			datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_sliding/'+miRNA+'_'+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])


"""keypoint functions"""


#Figure 6
def all_keypoint_regression_on_slope_and_mean(root_folder,overwrite=True,min_age=3*24,target='lifespan',bootstrap=False):
	
	#performs ordinary least squares regression on keypoint-annotated expression data using mean and slope. Expects z-scored data.
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if len(exp_dirs)>1:
			median_lifespans={}
			groups=worms.group_by([w.name.split()[0] for w in worms])	
			for exp, wormies in groups.items():
				median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
			for worm in worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
				lifespan_feature='scaled_lifespan'
				age_feature='scaled_age'
		else:
			lifespan_feature='lifespan'
			age_feature='age'
						
		keys=list(worms[10].td._keys())			
		lifespans=worms.get_feature(lifespan_feature)
		possible_max_ages=[]
		#not all timepoints are annotated, so infer what the best max age is 
		for worm in worms:
			test_data=getattr(worm.td,keys[10])
			test_ages=worm.td.age
			bad_ages=test_ages[numpy.isnan(test_data)]
			max_age=bad_ages[bad_ages>48]
			if len(max_age>0):
				max_age=max_age[0]/24
				possible_max_ages.append(max_age)
			else:
				max_age=math.ceil((numpy.percentile(lifespans/24,10)))
				possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age*24>numpy.percentile(lifespans,10):
				max_age=math.ceil(numpy.percentile(lifespans/24,10))	
		print(max_age)
	

		for key in list(worms[0].td._keys()):
			if 'z' in key and 'timepoint' not in key and 'age' not in key and 'centroid_dist' not in key and 'rms_dist' not in key and 'length' not in key and 'stage' not in key and 'max_width' not in key and 'surface_area' not in key and 'volume' not in key and 'projected_area' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
				filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > max_age*24)
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age*24,key,age_feature=age_feature),get_slope_wrapper(min_age,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
				slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
				average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
				if bootstrap==True:
					bootstrapped_correlations=[]
					bootstrapped_slope_correlations=[]
					bootstrapped_average_correlations=[]
					for i in range(0,1000):
						sample_worms=numpy.random.choice(filtered_worms,len(filtered_worms),replace=True)
						sample_worms=worm_data.Worms(sample_worms)	
						bootstrapped_results=sample_worms.regress(get_average_wrapper(min_age,max_age*24,key,age_feature=age_feature),get_slope_wrapper(min_age,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
						bootstrapped_correlations.append(bootstrapped_results.R2)
						bootstrapped_slope_results=sample_worms.regress(get_slope_wrapper(min_age,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
						bootstrapped_slope_correlations.append(bootstrapped_slope_results.R2)
						bootstrapped_average_results=sample_worms.regress(get_average_wrapper(min_age,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
						bootstrapped_average_correlations.append(bootstrapped_average_results.R2)	
			
					correlations_to_write.append([2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,97.5), 2*numpy.mean(bootstrapped_slope_correlations-numpy.percentile(bootstrapped_slope_correlations,2.5))])
					correlations_to_write.append([2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,97.5), 2*numpy.mean(bootstrapped_average_correlations-numpy.percentile(bootstrapped_average_correlations,2.5))])
					correlations_to_write.append([2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,97.5), 2*numpy.mean(bootstrapped_correlations-numpy.percentile(bootstrapped_correlations,2.5))])
					features_to_write.append(key+'_bootstrapped_slope')
					features_to_write.append(key+'_bootstrapped_mean')
					features_to_write.append(key+'_bootstrapped_joint')
				lifespans=results.y_est+results.resid			
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(lifespans,results.y_est)
				features_to_write.append('p_value')
				correlations_to_write.append(p_value)		
				color_vals = colorize.scale(lifespans/24, output_max=1)
				colors = colorize.color_map(color_vals, uint8=False)	
				plt.scatter(lifespans/24,results.y_est/24,c=colors)
				(m,b) = numpy.polyfit(lifespans/24, results.y_est/24, 1)
				yp=numpy.polyval([m,b], lifespans/24)
				mew, mewmew=zip(*sorted(zip(lifespans/24, yp)))
				mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
				plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
				ftext='R2 = '+(str)(round(results.R2,3))
				more_text="n= " + (str)(len(lifespans))
				plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
				ax = plt.gca() 
				ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
				ax.xaxis.set_label_coords(.85, .05)
				ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict,fontsize=24)
				ax.yaxis.set_label_coords(.05,.5)
				ax.set_xlim(left=100/24, right=450/24)
				plt.figtext(.15, .15, more_text, fontsize=24, ha='left')
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()
		features_to_write.append('n')
		features_to_write.append('max_age')
		correlations_to_write.append(len(lifespans))
		correlations_to_write.append(max_age)
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean/'+miRNA+'_keypoint_regression_correlations.tsv',[features_to_write]+[correlations_to_write])

#Figure 6
def all_keypoint_regression_on_slope_and_mean_plus_keypoint(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
	
	#Multivariable regression on each keypoint measure in combination with each of every other keypoint measure. Useful for seeing if any given keypoints are additive
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'

		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'			
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(lifespan_feature)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[10])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=math.ceil((numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age*24>numpy.percentile(lifespans,10):
				max_age=math.ceil(numpy.percentile(lifespans/24,10))	
		print(max_age)
		

		ages=[]
		
		for i in range(min_age,max_age+1):
			ages.append(i)	
		print(ages)
		keypoint_keys=list(worms[0].td._keys())
		for key in keypoint_keys:
			if 'z' in key and 'centroid_dist' not in key and 'rms_dist' not in key and 'length' not in key and 'stage' not in key and 'max_width' not in key and 'surface_area' not in key and 'volume' not in key and 'projected_area' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				
				for i in range(0, len(keypoint_keys)):
					if 'z' in keypoint_keys[i] and 'age' not in keypoint_keys[i] and 'stage' not in keypoint_keys[i] and 'timepoint' not in keypoint_keys[i] and 'timestamp' not in keypoint_keys[i]:
						save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_plus_keypoint/'+miRNA+'_'+key+'plus_'+keypoint_keys[i]+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean_plus_keypoint/'+miRNA+'_'+key+'plus_'+keypoint_keys[i]+'.svg'
						filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > max_age*24)
						uncontrolled_results=filtered_worms.regress(get_average_wrapper(min_age,max_age*24,key,age_feature=age_feature),get_slope_wrapper(min_age,max_age*24,key,age_feature=age_feature),get_average_wrapper(min_age,max_age*24,keypoint_keys[i],age_feature=age_feature),get_slope_wrapper(min_age,max_age*24,keypoint_keys[i],age_feature=age_feature),target=lifespan_feature)
						lifespans=uncontrolled_results.y_est+uncontrolled_results.resid	
						slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(lifespans,uncontrolled_results.y_est)
						ptext="p = " + (str)(round(p_value, 3))	
						color_vals = colorize.scale(lifespans/24, output_max=1)
						colors = colorize.color_map(color_vals, uint8=False)
						plt.scatter(lifespans/24,uncontrolled_results.y_est/24,c='indigo',alpha=.7)	
						(m,b) = numpy.polyfit(lifespans/24, uncontrolled_results.y_est/24, 1)
						yp=numpy.polyval([m,b], lifespans/24)
						mew, mewmew=zip(*sorted(zip(lifespans/24, yp)))
						mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
						plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='indigo',alpha=.7)
						gtext='R2 = '+(str)(round(uncontrolled_results.R2,3))
						more_text="n= " + (str)(len(lifespans))
						plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+gtext,fontdict=title_fontdict)
						ax = plt.gca() 
						ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
						ax.xaxis.set_label_coords(.85, .05)
						ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict,fontsize=24)
						ax.yaxis.set_label_coords(.05,.5)
						ax.set_xlim(left=100/24, right=450/24)
						plt.figtext(.15, .15, more_text, fontsize=24, ha='left')
						plt.figtext(.15,.7,gtext,color='black')
						plt.margins(.15,.15)
						print('Saving '+save_name)	
						plt.savefig(save_name)
						plt.savefig(save_name_box)
						plt.gcf().clf
						plt.close()
#Extra							
def all_keypoint_regression_on_slope_and_mean_controlled(root_folder,min_age=3,target='lifespan'):

	#performs semipartial correlation on each keypoint 'controlling' for each of every other keypoint. Again useful for seeing if things are indepdendent
	
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if len(exp_dirs)>1:
			median_lifespans={}
			groups=worms.group_by([w.name.split()[0] for w in worms])	
			for exp, wormies in groups.items():
				median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
			for worm in worms:
					exp = worm.name.split()[0]
					worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
					worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
					lifespan_feature='scaled_lifespan'
					age_feature='scaled_age'
		else:
			lifespan_feature='lifespan'
			age_feature='age'			
		keys=list(worms[10].td._keys())			
		lifespans=worms.get_feature(lifespan_feature)
		possible_max_ages=[]
		for worm in worms:
			test_data=getattr(worm.td,keys[10])
			test_ages=worm.td.age
			bad_ages=test_ages[numpy.isnan(test_data)]
			max_age=bad_ages[bad_ages>48]
			if len(max_age>0):
				max_age=max_age[0]/24
				possible_max_ages.append(max_age)
			else:
				#max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				max_age=math.ceil((numpy.percentile(lifespans/24,10)))
				possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age*24>numpy.percentile(lifespans,10):
				max_age=math.ceil(numpy.percentile(lifespans/24,10))	
		print(max_age)
		
		keypoint_keys=list(worms[0].td._keys())
		for key in keypoint_keys:
			try:
				if 'z' in key and 'age' not in key and 'stage' not in key and 'centroid_dist' not in key and 'timepoint' not in key and 'timestamp' not in key and 'surface_area' and 'max_width' not in key and 'volume' not in key and 'rms_dist' not in key and 'projected_area' not in key and 'length' not in key and 'centroid_dist' not in key and 'surface_area' not in key:
					miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
					if len(miRNA)>1:
						miRNA=miRNA[0]+'_'+miRNA[1]
					else:
						miRNA=miRNA[0]			
					for i in range(0, len(keypoint_keys)):
						if 'z' in key and 'age' not in keypoint_keys[i] and 'stage' not in keypoint_keys[i] and 'timepoint' not in keypoint_keys[i] and 'timestamp' not in keypoint_keys[i] and 'surface_area' not in keypoint_keys[i] and 'max_width' not in keypoint_keys[i] and 'volume' not in keypoint_keys[i] and 'rms_dist' not in keypoint_keys[i] and 'projected_area' not in keypoint_keys[i] and 'length' not in keypoint_keys[i] and 'centroid_dist' not in keypoint_keys[i]:
							save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_controlled/'+miRNA+'_'+key+'controlled_for_'+keypoint_keys[i]+'.svg'
							save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean_controlled/'+miRNA+'_'+key+'controlled_for_'+keypoint_keys[i]+'.svg'
							filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature)> max_age*24)
							uncontrolled_results=filtered_worms.regress(get_average_wrapper(min_age*24,max_age*24,key,age_feature=age_feature),get_slope_wrapper(min_age*24,max_age*24,key,age_feature=age_feature),target=lifespan_feature)
							results=filtered_worms.regress(get_average_wrapper(min_age*24,max_age*24,key,age_feature=age_feature),get_slope_wrapper(min_age*24,max_age*24,key,age_feature=age_feature),target=lifespan_feature,control_features=[get_average_wrapper(min_age*24,max_age*24,keypoint_keys[i],age_feature=age_feature),get_slope_wrapper(min_age*24,max_age*24,keypoint_keys[i],age_feature=age_feature)])
							lifespans=results.y_est+results.resid	
							slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(lifespans,results.y_est)
							ptext="p = " + (str)(round(p_value, 3))	
							color_vals = colorize.scale(lifespans/24, output_max=1)
							colors = colorize.color_map(color_vals, uint8=False)
							plt.scatter(lifespans/24,uncontrolled_results.y_est/24,c='indigo',alpha=.7)	
							plt.scatter(lifespans/24,results.y_est/24,c='gray',alpha=.7,marker='+')
							(m,b) = numpy.polyfit(lifespans/24, results.y_est/24, 1)
							yp=numpy.polyval([m,b], lifespans/24)
							mew, mewmew=zip(*sorted(zip(lifespans/24, yp)))
							mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
							plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
							(m,b) = numpy.polyfit(lifespans/24, uncontrolled_results.y_est/24, 1)
							yp=numpy.polyval([m,b], lifespans/24)
							mew, mewmew=zip(*sorted(zip(lifespans/24, yp)))
							mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
							plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='indigo',alpha=.7)
							ftext='R2 = '+(str)((results.R2))
							gtext='R2 = '+(str)((uncontrolled_results.R2))
							more_text="n= " + (str)(len(lifespans))
							plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
							ax = plt.gca() 
							ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
							ax.xaxis.set_label_coords(.85, .05)
							ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict,fontsize=24)
							ax.yaxis.set_label_coords(.05,.5)
							ax.set_xlim(left=100/24, right=450/24)
							plt.figtext(.15, .15, more_text, fontsize=24, ha='left')
							plt.figtext(.15,.8,ftext,color='gray')
							plt.figtext(.15,.7,gtext,color='black')
							plt.margins(.15,.15)
							print('Saving '+save_name)	
							plt.savefig(save_name)
							plt.savefig(save_name_box)
							plt.gcf().clf
							plt.close()
			except:
				continue				
#Extra
def all_keypoint_regression_on_slope_and_mean_sliding(root_folder,min_age=3*24,target='lifespan', bootstrap=True):

#regression on keypoint data using sliding window. 

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))
			lifespan_feature='lifespan'
			age_feature='age'
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
			else:
				lifespan_feature='lifespan'
				age_feature='age'			
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(lifespan_feature)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[10])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
				max_age=math.ceil((numpy.percentile(possible_max_ages,50)))
				if max_age*24>numpy.percentile(lifespans,10):
					max_age=math.ceil(numpy.percentile(lifespans/24,10))	
			print(max_age)
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
			if len(miRNA)>1:
				miRNA=miRNA[0]+'_'+miRNA[1]
			else:
				miRNA=miRNA[0]
		keypoint_keys=list(worms[10].td._keys())		
		for key in keypoint_keys:
			if 'z' in key and 'centroid_dist' not in key and 'rms_dist' not in key and 'length' not in key and 'stage' not in key and 'max_width' not in key and 'surface_area' not in key and 'volume' not in key and 'projected_area' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				ages=[]
				correlations=[]
				windows=[]
				for i in range(min_age,(int)(max_age*24)+(int)(max_age*24)%12,12):
					ages.append(i)		
				for i in range(0, len(ages)-1):
					left_window=ages[i]
					for j in range(ages.index(left_window)+1, len(ages)):
						right_window=ages[j]
						filtered_worms=worms.filter(lambda worm: getattr(worm,lifespan_feature) > right_window)
						filtered_lifespans=filtered_worms.get_feature(lifespan_feature)
						try:	
							results=filtered_worms.regress(get_average_wrapper(left_window,right_window,key,age_feature=age_feature),get_slope_wrapper(left_window,right_window,key,age_feature=age_feature),target=lifespan_feature)
						except:
							continue
						correlations.append(results.R2)
						windows.append([left_window,right_window])		
				
				best_window=windows[correlations.index(max(correlations))]
				print(best_window)
				results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),get_slope_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),target=lifespan_feature)				
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(results.y_est+results.resid,results.y_est)
				ptext="p = " + (str)(round(p_value, 3))	
				color_vals = colorize.scale((results.y_est+results.resid)/24, output_max=1)
				colors = colorize.color_map(color_vals, uint8=False)
				plt.scatter((results.y_est+results.resid)/24,results.y_est/24,c=colors)			
				(m,b) = numpy.polyfit((results.y_est+results.resid)/24, results.y_est/24, 1)
				yp=numpy.polyval([m,b], (results.y_est+results.resid)/24)
				mew, mewmew=zip(*sorted(zip((results.y_est+results.resid)/24, yp)))
				mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
				plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
				slope_results=filtered_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),target=lifespan_feature)
				average_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),target=lifespan_feature)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
				if bootstrap==True:
					bootstrapped_correlations=[]
					bootstrapped_slope_correlations=[]
					bootstrapped_average_correlations=[]
				for i in range(0,1000):
					sample_worms=numpy.random.choice(filtered_worms,len(filtered_worms),replace=True)
					sample_worms=worm_data.Worms(sample_worms)	
					bootstrapped_results=sample_worms.regress(get_average_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),get_slope_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),target=lifespan_feature)
					bootstrapped_correlations.append(bootstrapped_results.R2)
					bootstrapped_slope_results=sample_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),target=lifespan_feature)
					bootstrapped_slope_correlations.append(bootstrapped_slope_results.R2)
					bootstrapped_average_results=sample_worms.regress(get_average_wrapper(best_window[0],best_window[1],key,age_feature=age_feature),target=lifespan_feature)
					bootstrapped_average_correlations.append(bootstrapped_average_results.R2)	
			
				correlations_to_write.append([2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,97.5), 2*numpy.mean(bootstrapped_slope_correlations)-numpy.percentile(bootstrapped_slope_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,97.5), 2*numpy.mean(bootstrapped_average_correlations)-numpy.percentile(bootstrapped_average_correlations,2.5)])
				correlations_to_write.append([2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,97.5), 2*numpy.mean(bootstrapped_correlations)-numpy.percentile(bootstrapped_correlations,2.5)])
				features_to_write.append(key+'_bootstrapped_slope')
				features_to_write.append(key+'_bootstrapped_mean')
				features_to_write.append(key+'_bootstrapped_joint')
				features_to_write.append('best_window')
				correlations_to_write.append(best_window)
				features_to_write.append('n')
				correlations_to_write.append(len(results.y_est))
				gtext='R2 = '+(str)(round(results.R2,3))
				more_text="n= " + (str)(len(filtered_lifespans))
				plt.title('P'+miRNA+'::GFP vs. lifespan: '+gtext,fontdict=title_fontdict)
				ax = plt.gca() 
				ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
				ax.xaxis.set_label_coords(.85, .05)
				ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict,fontsize=24)
				ax.yaxis.set_label_coords(.05,.5)
				ax.set_xlim(left=100/24, right=450/24)
				plt.figtext(.15, .15, more_text, fontsize=24, ha='left')
				plt.figtext(.15,.7,gtext,color='black')
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.gcf().clf
				plt.close()
		rows=[[] for _ in range(len(features_to_write))]
		for col in features_to_write:	
			for row, colval in zip(rows,correlations_to_write):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_sliding/'+miRNA+'_keypoint_regression_correlations.tsv',[features_to_write]+[correlations_to_write])		

#Figure 6 
def all_keypoint_population_trajectory_cohorts(root_folder,min_age=48,time_units='days'):
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
			age_feature='age'
			gfp_feature=key
			lifespan_feature=lifespan
		else:
			max_age=numpy.inf		
		for key in list(worms[0].td._keys()):
			if 'age' not in key and 'timepoint' not in key and 'timestamp' not in key and 'z' not in key and 'centroid_dist' not in key:
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
				try:
					worms.get_time_range(key,24,numpy.inf)
				except:
					continue				
				if len(exp_dirs)>1:
					median_gfp={}
					median_lifespans={}
					groups=worms.group_by([w.name.split()[0] for w in worms])	
	
					for exp, wormies in groups.items():
							data = wormies.get_time_range(key,min_age,max_age)	
							median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
							median_gfp[exp] = numpy.median([d[:,1].mean() for d in data])
					for worm in worms:
							exp = worm.name.split()[0]
							worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
							worm.scaled_lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.scaled_age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
							lifespan_feature='scaled_lifespan'
							gfp_feature='scaled_gfp'
							age_feature='scaled_age'				
				else:
					lifespan_feature='lifespan'
					gfp_feature=key
					age_feature='age'
				lifespans=worms.get_feature(lifespan_feature)
				bins=worms.bin(lifespan_feature,nbins=5, equal_count=True)
				averaged_worms = worm_data.meta_worms(bins, gfp_feature, age_feature=age_feature)
				figure=averaged_worms.plot_timecourse(gfp_feature,age_feature=age_feature,min_age=min_age,max_age=max_age,time_units='days')	
				cohorts=[]
				for mew, mewmew in bins.items():
					cohorts.append(len(mewmew))	
				text='n = ' + (str)(cohorts)
				plt.figtext(.5, .15, text, fontdict=text_fontdict, fontsize=28)
				plt.title('P'+miRNA+'::GFP',fontdict=title_fontdict, loc='left')
				ax = plt.gca() 
				ax.set_xlabel('Days', fontdict=label_fontdict)
				ax.xaxis.set_label_coords(.95, .05)
				ax.set_ylabel('Expression ('+key+')',fontdict=label_fontdict, fontsize=24)
				ax.yaxis.set_label_coords(.05,.5)
				ax.set_xlim(left=-.5,right=19)
				plt.ylabel('P'+miRNA+'::GFP',fontdict=label_fontdict, fontsize=24)
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()	

"""miRNA mutant functions"""


#Figure 4
def all_mutant_regression_on_slope_and_mean(root_folder,min_age=3,target='lifespan',bootstrap=False):
	#Regression on slope and mean of expression, fixed time window
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])		
		if len(mut_exp_dirs)>1:
			mut_median_lifespans={}
			mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
			control_median_lifespans={}
			control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
			for exp, wormies in control_groups.items():
				control_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
			for exp, wormies in mut_groups.items():
				mut_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
			for worm in mut_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
				age_feature='scaled_age'
				lifespan_feature='scaled_lifespan'
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]
				age_feature='scaled_age'
				lifespan_feature='scaled_lifespan'	

		mut_lifespans=mut_worms.get_feature(lifespan_feature)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans,10)))
		control_lifespans=control_worms.get_feature(lifespan_feature)
		control_max_age=(int)(round(numpy.percentile(control_lifespans,10)))	
		mut_features_to_write=[]
		control_features_to_write=[]
		mut_correlations_to_write=[]
		control_correlations_to_write=[]
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_mutant_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_mutant_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'	
			filtered_mut_worms=mut_worms.filter(lambda worm: getattr(worm,lifespan_feature) > mut_max_age)
			filtered_control_worms=control_worms.filter(lambda worm: getattr(worm,lifespan_feature) > control_max_age)
			mut_results=filtered_mut_worms.regress(get_average_wrapper(min_age*24,mut_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age*24,mut_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			control_results=filtered_control_worms.regress(get_average_wrapper(min_age*24,control_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age*24,control_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			mut_features_to_write.append(key)
			control_features_to_write.append(key)
			mut_correlations_to_write.append(mut_results.R2)
			control_correlations_to_write.append(control_results.R2)
			if bootstrap==True:
				bootstrapped_mut_correlations=[]
				bootstrapped_control_correlations=[]
				for i in range(0,1000):
					sample_mut_worms=numpy.random.choice(filtered_mut_worms,len(filtered_mut_worms),replace=True)
					sample_mut_worms=worm_data.Worms(sample_mut_worms)
					sample_control_worms=numpy.random.choice(filtered_control_worms,len(filtered_control_worms),replace=True)
					sample_control_worms=worm_data.Worms(sample_control_worms)		
					bootstrapped_mut_results=sample_mut_worms.regress(get_average_wrapper(min_age*24,mut_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age*24,mut_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_mut_correlations.append(bootstrapped_mut_results.R2)
					bootstrapped_control_results=sample_control_worms.regress(get_average_wrapper(min_age*24,control_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age*24,control_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_control_correlations.append(bootstrapped_control_results.R2)
			
				mut_correlations_to_write.append([2*numpy.mean(bootstrapped_mut_correlations)-numpy.percentile(bootstrapped_mut_correlations,97.5), 2*numpy.mean(bootstrapped_mut_correlations)-numpy.percentile(bootstrapped_mut_correlations,2.5)])
				mut_features_to_write.append(key+'_bootstrapped_joint')
				control_correlations_to_write.append([2*numpy.mean(bootstrapped_control_correlations)-numpy.percentile(bootstrapped_control_correlations,97.5), 2*numpy.mean(bootstrapped_control_correlations)-numpy.percentile(bootstrapped_control_correlations,2.5)])
				control_features_to_write.append(key+'_bootstrapped_joint')
			mut_features_to_write.append('max_age')
			mut_correlations_to_write.append(mut_max_age)
			control_features_to_write.append('max_age')
			control_correlations_to_write.append(control_max_age)
			mut_lifespans=filtered_mut_worms.get_feature(lifespan_feature)
			control_lifespans=filtered_control_worms.get_feature(lifespan_feature)	
			orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
			plt.legend(handles=[orangered_patch,indigo_patch])
			slope, intercept, r_value, mut_p_value, std_err=scipy.stats.linregress(mut_lifespans,mut_results.y_est)
			slope, intercept, r_value, control_p_value, std_err=scipy.stats.linregress(control_lifespans,control_results.y_est)
			plt.scatter(mut_lifespans/24,mut_results.y_est/24,c='orangered',marker='+',s=50,edgecolor='orangered',alpha=.7)
			plt.scatter(control_lifespans/24,control_results.y_est/24,c='indigo',marker='o',s=50,edgecolor='indigo',alpha=.7)
			(m,b) = numpy.polyfit(mut_lifespans/24, mut_results.y_est/24, 1)
			yp=numpy.polyval([m,b], mut_lifespans/24)
			mew, mewmew=zip(*sorted(zip(mut_lifespans/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='orangered',alpha=.7)
			(m,b) = numpy.polyfit(control_lifespans/24, control_results.y_est/24, 1)
			yp=numpy.polyval([m,b], control_lifespans/24)
			mew, mewmew=zip(*sorted(zip(control_lifespans/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='indigo',alpha=.7)
			plt.title('P' +miRNA+'::GFP expression vs. lifespan',fontdict=title_fontdict)
			plt.xlabel('Lifespan (days)',fontdict=label_fontdict, fontsize=24)
			plt.ylabel("Expression-predicted lifespan (days)",fontdict=label_fontdict, fontsize=24)
			mut_ftext="r2 = "+(str)(round(mut_results.R2,3))
			control_ftext="r2 = "+(str)(round(control_results.R2,3))
			plt.title('P' +miRNA+'::GFP expression vs. lifespan',fontdict=title_fontdict)
			mut_more_text="n= " + (str)(len(mut_lifespans))
			control_more_text="n= " + (str)(len(control_lifespans))
			plt.figtext(.15,.8,mut_ftext,fontsize=20,ha='left',color='orangered')
			plt.figtext(.15,.7,control_ftext,fontsize=20,ha='left',color='indigo')
			plt.figtext(.8, .2, mut_more_text, fontsize=20, ha='left',color='orangered')
			plt.figtext(.8, .15, control_more_text, fontsize=20, ha='left',color='indigo')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
		rows=[[] for _ in range(len(mut_features_to_write))]
		for col in mut_features_to_write:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_mutant_regression_on_slope_and_mean/'+miRNA+'_'+'mutant_correlations.tsv',[mut_features_to_write]+[mut_correlations_to_write])
		
		rows=[[] for _ in range(len(control_features_to_write))]
		for col in control_features_to_write:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_mutant_regression_on_slope_and_mean/'+miRNA+'_'+'control_correlations.tsv',[control_features_to_write]+[control_correlations_to_write])					

#Figure 4
def all_mutant_correlation_plots(root_folder,min_age=2,target='lifespan'):
	#correlation plots for mutants. Worm must be alive entire 24 hour time period 
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])			
		if len(mut_exp_dirs)>1:
			mut_median_lifespans={}
			mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
			control_median_lifespans={}
			control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
			for exp, wormies in control_groups.items():
				control_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
			for exp, wormies in mut_groups.items():
				mut_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
			for worm in mut_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
				age_feature='scaled_age'
				lifespan_feature='scaled_lifespan'
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]
				age_feature='scaled_age'
				lifespan_feature='scaled_lifespan'	

		mut_lifespans=mut_worms.get_feature(lifespan_feature)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans/24,10)))
		mut_extended_max_age=(int)(round(numpy.percentile(mut_lifespans/24,75)))
		control_lifespans=control_worms.get_feature(lifespan_feature)
		control_max_age=(int)(round(numpy.percentile(control_lifespans/24,10)))
		control_extended_max_age=(int)(round(numpy.percentile(control_lifespans/24,75)))
		mut_ages=[]
		control_ages=[]	
		for i in range(min_age,mut_extended_max_age+1):
			mut_ages.append(i)
		for i in range(min_age,control_extended_max_age+1):
			control_ages.append(i)			
		features_towrite=['age']
		mut_correlations_towrite=[mut_ages[1::]]
		control_correlations_towrite=[control_ages[1::]]
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_mutant_correlation_plots/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_mutant_correlation_plots/'+miRNA+'_'+key+'.svg'
			mut_correlations=[]
			control_correlations=[]
			mut_p_values=[]
			control_p_values=[]
			mut_n=[]
			control_n=[]	
			for i in range(0,len(mut_ages)-1):
				filtered_mut_worms=mut_worms.filter(lambda worm: getattr(worm,lifespan_feature) > mut_ages[i+1]*24)
				mut_results=filtered_mut_worms.regress(get_average_wrapper(mut_ages[i]*24,mut_ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(mut_ages[i]*24,mut_ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
				mut_correlations.append(mut_results.R2)
				filtered_mut_lifespans=filtered_mut_worms.get_feature(lifespan_feature)
				slope, intercept,rvalue,mut_pvalue,stderror=scipy.stats.linregress(filtered_mut_lifespans,mut_results.y_est)
				mut_p_values.append(mut_pvalue)
				mut_n.append(len(filtered_mut_worms))
			for i in range(0,len(control_ages)-1):
				filtered_control_worms=control_worms.filter(lambda worm: getattr(worm,lifespan_feature) > control_ages[i+1]*24)
				control_results=filtered_control_worms.regress(get_average_wrapper(control_ages[i]*24,control_ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(control_ages[i]*24,control_ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
				control_correlations.append(control_results.R2)
				filtered_control_lifespans=filtered_control_worms.get_feature(lifespan_feature)
				slope, intercept,rvalue,control_pvalue,stderror=scipy.stats.linregress(filtered_control_lifespans,control_results.y_est)
				control_p_values.append(control_pvalue)	
				control_n.append(len(filtered_control_worms))
			mut_correlations_towrite.append(mut_correlations)
			control_correlations_towrite.append(control_correlations)
			features_towrite.append(key)
			features_towrite.append('p_value')
			mut_correlations_towrite.append(mut_p_values)
			control_correlations_towrite.append(control_p_values)
			features_towrite.append('n')
			mut_correlations_towrite.append(mut_n)
			control_correlations_towrite.append(control_n)
			mut_p_x=[]
			mut_p_y=[]
			for i in range(0,len(mut_p_values)):
				if mut_p_values[i]<.05 and mut_correlations[i]>=.1:
					mut_p_y.append(mut_correlations[i]+.03)
					mut_p_x.append(mut_ages[1::][i])
			control_p_x=[]
			control_p_y=[]
			for i in range(0,len(control_p_values)):
				if control_p_values[i]<.05 and control_correlations[i]>=.1:
					control_p_y.append(control_correlations[i]+.03)
					control_p_x.append(control_ages[1::][i])

			orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
			plt.legend(handles=[orangered_patch,indigo_patch])
			plt.scatter(mut_p_x,mut_p_y,marker=(6,2,0),c='orangered',s=50,alpha=.7)
			plt.scatter(control_p_x,control_p_y,marker=(6,2,0),c='indigo',s=50,alpha=.7)
			plt.scatter(mut_ages[1::],mut_correlations,c='orangered',marker='o',s=50,edgecolor='orangered',alpha=.7)
			plt.scatter(control_ages[1::],control_correlations,c='indigo',marker='o',s=50,edgecolor='indigo',alpha=.7)
			plt.plot(mut_ages[1:mut_ages.index(mut_max_age)+1],mut_correlations[0:mut_ages.index(mut_max_age)], c='orangered',alpha=.7)
			plt.plot(control_ages[1:control_ages.index(control_max_age)+1],control_correlations[0:control_ages.index(control_max_age)], c='indigo',alpha=.7)
			plt.plot(mut_ages[mut_ages.index(mut_max_age)::],mut_correlations[mut_ages.index(mut_max_age)-1::],c='orangered',linestyle='--',alpha=.7)
			plt.plot(control_ages[control_ages.index(control_max_age)::],control_correlations[control_ages.index(control_max_age)-1::],c='indigo',linestyle='--',alpha=.7)	
			plt.title('P'+miRNA + ' lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
			plt.xlabel('Age (day post-hatch)', fontdict=label_fontdict, fontsize=24)
			plt.ylabel('Coefficient of determination (r2)',fontdict=label_fontdict, fontsize=24)
			plt.ylim(0,.6)
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
		n=len(mut_ages)
		rows=[[] for _ in range(n)]
		for col in mut_correlations_towrite:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_mutant_correlation_plots/'+miRNA+'_'+'mutant_correlations.tsv',[features_towrite]+rows)
		n=len(control_ages)
		rows=[[] for _ in range(n)]
		for col in control_correlations_towrite:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_mutant_correlation_plots/'+miRNA+'_'+'control_correlations.tsv',[features_towrite]+rows)

#Figure 4
def all_mutant_survival_curves (root_folder):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])
		miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
		mut_lifespans=mut_worms.get_feature('lifespan')/24
		mut_max_life=mut_lifespans.max()
		mut_min_life=mut_lifespans.min()
		mut_days=numpy.arange(3,mut_max_life+1,.25)
		control_lifespans=control_worms.get_feature('lifespan')/24
		control_max_life=control_lifespans.max()
		control_min_life=control_lifespans.min()
		control_days=numpy.arange(3,control_max_life+1,.25)
		mut_percent_alive = []
		control_percent_alive=[]
		for i in mut_days:
			count =0
			for item in mut_lifespans:
				if item > i:
					count=count+1
			mut_percent_alive.append((count/len(mut_lifespans))*100)
		for i in control_days:
			count =0
			for item in control_lifespans:
				if item > i:
					count=count+1
			control_percent_alive.append((count/len(control_lifespans))*100)	
		orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
		indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
		plt.legend(handles=[orangered_patch,indigo_patch])
		plt.plot(mut_days,mut_percent_alive,color='orangered',alpha=.3)
		plt.plot(control_days,control_percent_alive,color='indigo',alpha=.3)	
		plt.scatter(mut_days,mut_percent_alive,color='orangered')
		plt.scatter(control_days,control_percent_alive,color='indigo')
		plt.xlabel("Age (days)",fontdict=label_fontdict)
		plt.ylabel("Survival (%)",fontdict=label_fontdict)
		more_text="median mutant lifespan = " + (str)((round(numpy.median(mut_lifespans),1))) + " days"
		more_text_ctr="median control lifespan = " + (str)((round(numpy.median(control_lifespans),1))) + " days"
		even_more_text="n = "+(str)(len(mut_lifespans))
		even_more_text_ctr="n = "+(str)(len(control_lifespans))
		plt.figtext(.15, .25, more_text, fontsize=15, ha='left')
		plt.figtext(.15, .22, more_text_ctr, fontsize=15, ha='left')
		plt.figtext(.15, .19, even_more_text, fontsize=15, ha='left')
		plt.figtext(.15, .16, even_more_text_ctr, fontsize=15, ha='left')
		save_name='/Volumes/9karray/Kinser_Holly/all_mutant_survival_curves/'+miRNA+'_survival_curve.svg'
		save_name_box='/Users/pincuslab/Box/miRNA Data/all_mutant_survival_curves/'+miRNA+'_survival_curve.svg'
		plt.savefig(save_name)
		plt.savefig(save_name_box)
		plt.gcf().clf
		plt.close()
#Figure 4
def all_mutant_population_trajectory(root_folder,min_age=24,time_units='hours'):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			try:
				mut_worms.get_time_range(key,24,numpy.inf)
			except:
				continue
			max_age=numpy.inf				
			if len(mut_exp_dirs)>1:
				mut_median_gfp={}
				mut_median_lifespans={}
				control_median_gfp={}
				control_median_lifespans={}
				mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
				control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
				for exp, wormies in mut_groups.items():
						mut_data = wormies.get_time_range(key,min_age,max_age)
						mut_median_gfp[exp] = numpy.median([numpy.median(d[:,1]) for d in mut_data])
						mut_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for exp, wormies in control_groups.items():
						control_data = wormies.get_time_range(key,min_age,max_age)			
						control_median_gfp[exp] = numpy.median([numpy.median(d[:,1]) for d in control_data])
						control_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in mut_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ mut_median_gfp[exp] * mut_median_gfp[mut_exp_dirs[0]]
						worm.scaled_lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
				for worm in control_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ control_median_gfp[exp] * control_median_gfp[control_exp_dirs[0]]
						worm.scaled_lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'				

			mut_lifespans=mut_worms.get_feature(lifespan_feature)
			control_lifespans=control_worms.get_feature(lifespan_feature)
			mut_bins={'['+(str)(mut_lifespans.min())+'-'+(str)(mut_lifespans.max())+']':mut_worms}
			control_bins={'['+(str)(control_lifespans.min())+'-'+(str)(control_lifespans.max())+']':control_worms}
			mut_averaged_worms = worm_data.meta_worms(mut_bins, 'scaled_gfp', age_feature=age_feature)
			control_averaged_worms = worm_data.meta_worms(control_bins, 'scaled_gfp', age_feature=age_feature)		
			mut_cohorts=[]
			control_cohorts=[]
			for mew, mewmew in mut_bins.items():
				mut_cohorts.append(len(mewmew))	
			for mew, mewmew in control_bins.items():
				control_cohorts.append(len(mewmew))		
			mut_time_ranges=mut_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,age_feature)
			control_time_ranges=control_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,age_feature)
			mut_out=[]
			control_out=[]
			for time_range in mut_time_ranges:
				x,y=time_range.T
				mut_out.append((x,y))
			for time_range in control_time_ranges:
				x,y=time_range.T
				control_out.append((x,y))
			for x, y in mut_out:
				plt.plot(x/24,y, c='orangered',alpha=.7)
			for x, y in control_out:
				plt.plot(x/24,y, c='indigo',alpha=.7) 				
			orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
			plt.legend(handles=[orangered_patch,indigo_patch])
			mut_text='n = ' + (str)(mut_cohorts)
			control_text='n = ' + (str)(control_cohorts)
			plt.figtext(.5, .15, mut_text, fontdict=text_fontdict,fontsize=24)
			plt.figtext(.5, .2, control_text, fontdict=text_fontdict,fontsize=24)
			plt.title('P'+miRNA+'::GFP expression',fontdict=title_fontdict)
			plt.ylabel('Expression ('+value+')',fontdict=label_fontdict, fontsize=24)
			plt.margins(.15,.15)
			save_name='/Volumes/9karray/Kinser_Holly/all_mutant_population_trajectory/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_mutant_population_trajectory/'+miRNA+'_'+key+'.svg'
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.clf()
			plt.close()

"""daf 16 functions"""

#Figure 5
def all_daf16_population_trajectory(root_folder,min_age=24,time_units='hours'):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])
		max_age=numpy.inf		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[1]
			try:
				mut_worms.get_time_range(key,24,numpy.inf)
			except:
				continue		
			if len(mut_exp_dirs)>1:
				mut_median_gfp={}
				control_median_gfp={}
				mut_median_lifespans={}
				control_median_lifespans={}
				mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
				control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		

				for exp, wormies in mut_groups.items():
						mut_data = wormies.get_time_range(key,min_age,max_age)
						mut_median_gfp[exp] = numpy.median([d[:,1].mean() for d in mut_data])
						mut_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for exp, wormies in control_groups.items():
						control_data = wormies.get_time_range(key,min_age,max_age)			
						control_median_gfp[exp] = numpy.median([d[:,1].mean() for d in control_data])
						control_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in mut_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ mut_median_gfp[exp] * mut_median_gfp[mut_exp_dirs[0]]
						worm.scaled_lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'
				for worm in control_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ control_median_gfp[exp] * control_median_gfp[control_exp_dirs[0]]
						worm.scaled_lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
						worm.td.scaled_age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]
						lifespan_feature='scaled_lifespan'
						age_feature='scaled_age'

			mut_lifespans=mut_worms.get_feature(lifespan_feature)
			control_lifespans=control_worms.get_feature(lifespan_feature)
			mut_bins={'['+(str)(mut_lifespans.min())+'-'+(str)(mut_lifespans.max())+']':mut_worms}
			control_bins={'['+(str)(control_lifespans.min())+'-'+(str)(control_lifespans.max())+']':control_worms}
			mut_averaged_worms = worm_data.meta_worms(mut_bins, 'scaled_gfp', age_feature=age_feature)
			control_averaged_worms = worm_data.meta_worms(control_bins, 'scaled_gfp', age_feature=age_feature)
			mut_cohorts=[]
			control_cohorts=[]
			for mew, mewmew in mut_bins.items():
				mut_cohorts.append(len(mewmew))	
			for mew, mewmew in control_bins.items():
				control_cohorts.append(len(mewmew))		
			mut_time_ranges=mut_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,age_feature)
			control_time_ranges=control_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,age_feature)
			mut_out=[]
			control_out=[]
			for time_range in mut_time_ranges:
				x,y=time_range.T
				mut_out.append((x,y))
			for time_range in control_time_ranges:
				x,y=time_range.T
				control_out.append((x,y))
			for x, y in mut_out:
				plt.plot(x/24,y, c='orangered')
			for x, y in control_out:
				plt.plot(x/24,y, c='indigo') 				
			orangered_patch = mpatches.Patch(color='orangered', label='daf-16')
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type')
			plt.legend(handles=[orangered_patch,indigo_patch])
			mut_text='n = ' + (str)(mut_cohorts)
			control_text='n = ' + (str)(control_cohorts)
			plt.figtext(.5, .15, mut_text, fontdict=text_fontdict,color='orangered')
			plt.figtext(.5, .2, control_text, fontdict=text_fontdict,color='indigo')
			plt.title('P'+miRNA+'::GFP expression',fontdict=title_fontdict)
			plt.ylabel('Expression ('+value+')',fontdict=label_fontdict, fontsize=24)
			plt.xlabel('Days',fontdict=label_fontdict,fontsize=24)
			plt.margins(.15,.15)
			save_name='/Volumes/9karray/Kinser_Holly/all_daf16_population_trajectory/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_daf16_population_trajectory/'+miRNA+'_'+key+'.svg'
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.clf()
			plt.close()
#Figure 5
def all_daf16_regression_on_slope_and_mean(root_folder,min_age=3*24,target='lifespan',bootstrap=False):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])		
		if len(mut_exp_dirs)>1:
			mut_median_lifespans={}
			mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
			control_median_lifespans={}
			control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
			for exp, wormies in control_groups.items():
				control_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
			for exp, wormies in mut_groups.items():
				mut_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
			for worm in mut_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
				lifespan_feature='scaled_lifespan'
				age_feature='scaled_age'
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]
				lifespan_feature='scaled_lifespan'
				age_feature='scaled_age'
		else:
			lifespan_feature='lifespan'
			age_feature='age'			
		mut_lifespans=mut_worms.get_feature(lifespan_feature)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans,10)))
		control_lifespans=control_worms.get_feature(lifespan_feature)
		control_max_age=(int)(round(numpy.percentile(control_lifespans,10)))
		mut_features_to_write=[]
		control_features_to_write=[]
		mut_correlations_to_write=[]
		control_correlations_to_write=[]
		control_features_to_write.append('max_age')
		mut_features_to_write.append('max_age')
		control_correlations_to_write.append(control_max_age/24)
		mut_correlations_to_write.append(mut_max_age/24)
		for key,value in gfp_measures.items():
			try:
				mut_worms.get_time_range(key,24,numpy.inf)
			except:
				continue
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[1]
			save_name='/Volumes/9karray/Kinser_Holly/all_daf16_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_daf16_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			filtered_mut_worms=mut_worms.filter(lambda worm: getattr(worm,lifespan_feature) > mut_max_age)
			filtered_control_worms=control_worms.filter(lambda worm: getattr(worm,lifespan_feature) > control_max_age)
			mut_results=filtered_mut_worms.regress(get_average_wrapper(min_age,mut_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,mut_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			control_results=filtered_control_worms.regress(get_average_wrapper(min_age,control_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,control_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
			mut_lifespans=filtered_mut_worms.get_feature(lifespan_feature)
			mut_features_to_write.append(key)
			control_features_to_write.append(key)
			mut_correlations_to_write.append(mut_results.R2)
			control_correlations_to_write.append(control_results.R2)
			control_lifespans=filtered_control_worms.get_feature(lifespan_feature)	
			orangered_patch = mpatches.Patch(color='orangered', label='daf-16')
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type')
			plt.legend(handles=[orangered_patch,indigo_patch])
			slope, intercept, r_value, mut_p_value, std_err=scipy.stats.linregress(mut_lifespans,mut_results.y_est)
			slope, intercept, r_value, control_p_value, std_err=scipy.stats.linregress(control_lifespans,control_results.y_est)
			if bootstrap==True:
				bootstrapped_mut_correlations=[]
				bootstrapped_control_correlations=[]
				for i in range(0,1000):
					sample_mut_worms=numpy.random.choice(filtered_mut_worms,len(filtered_mut_worms),replace=True)
					sample_mut_worms=worm_data.Worms(sample_mut_worms)
					sample_control_worms=numpy.random.choice(filtered_control_worms,len(filtered_control_worms),replace=True)
					sample_control_worms=worm_data.Worms(sample_control_worms)		
					bootstrapped_mut_results=sample_mut_worms.regress(get_average_wrapper(min_age,mut_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,mut_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_mut_correlations.append(bootstrapped_mut_results.R2)
					bootstrapped_control_results=sample_control_worms.regress(get_average_wrapper(min_age,control_max_age,key+'_z',age_feature=age_feature),get_slope_wrapper(min_age,control_max_age,key+'_z',age_feature=age_feature),target=lifespan_feature)
					bootstrapped_control_correlations.append(bootstrapped_control_results.R2)
			
				mut_correlations_to_write.append([2*numpy.mean(bootstrapped_mut_correlations)-numpy.percentile(bootstrapped_mut_correlations,97.5), 2*numpy.mean(bootstrapped_mut_correlations)-numpy.percentile(bootstrapped_mut_correlations,2.5)])
				mut_features_to_write.append(key+'_bootstrapped_joint')
				control_correlations_to_write.append([2*numpy.mean(bootstrapped_control_correlations)-numpy.percentile(bootstrapped_control_correlations,97.5), 2*numpy.mean(bootstrapped_control_correlations)-numpy.percentile(bootstrapped_control_correlations,2.5)])
				control_features_to_write.append(key+'_bootstrapped_joint')
			plt.scatter(mut_lifespans/24,mut_results.y_est/24,c='orangered',marker='+',s=50,edgecolor='orangered',alpha=.5)
			plt.scatter(control_lifespans/24,control_results.y_est/24,c='indigo',marker='o',s=50,edgecolor='indigo',alpha=.5)
			(m,b) = numpy.polyfit(mut_lifespans/24, mut_results.y_est/24, 1)
			yp=numpy.polyval([m,b], mut_lifespans/24)
			mew, mewmew=zip(*sorted(zip(mut_lifespans/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='orangered',alpha=.7)
			(m,b) = numpy.polyfit(control_lifespans/24, control_results.y_est/24, 1)
			yp=numpy.polyval([m,b], control_lifespans/24)
			mew, mewmew=zip(*sorted(zip(control_lifespans/24, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='indigo',alpha=.7)
			plt.title('P' +miRNA+'::GFP expression vs. lifespan: R2 = '+(str)(round((mut_results.R2),2))+', '+(str)(round((control_results.R2),2)),fontdict=title_fontdict)
			plt.xlabel('Lifespan (days)',fontdict=label_fontdict)
			plt.ylabel("Expression-predicted lifespan (days)",fontdict=label_fontdict,fontsize=24)
			mut_more_text="n= " + (str)(len(mut_lifespans))
			control_more_text="n= " + (str)(len(control_lifespans))
			plt.figtext(.8, .2, mut_more_text, fontsize=20, ha='left',color='orangered')
			plt.figtext(.8, .15, control_more_text, fontsize=20, ha='left',color='indigo')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()	
		rows=[[] for _ in range(len(mut_features_to_write))]
		for col in mut_features_to_write:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_daf16_regression_on_slope_and_mean/'+miRNA+'_'+'mutant_correlations.tsv',[mut_features_to_write]+[mut_correlations_to_write])
		
		rows=[[] for _ in range(len(control_features_to_write))]
		for col in control_features_to_write:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_daf16_regression_on_slope_and_mean/'+miRNA+'_'+'control_correlations.tsv',[control_features_to_write]+[control_correlations_to_write])

#Figure 5
def all_daf16_correlation_plots(root_folder,min_age=2,target='lifespan'):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])			
		if len(mut_exp_dirs)>1:
			mut_median_lifespans={}
			mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
			control_median_lifespans={}
			control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
			for exp, wormies in control_groups.items():
				control_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
			for exp, wormies in mut_groups.items():
				mut_median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
			for worm in mut_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
				age_feature='scaled_age'
				lifespan_feature='scaled_lifespan'
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.scaled_lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.scaled_age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]
				age_feature='scaled_age'
				lifespan_feature='scaled_lifespan'
		else:
			lifespan_feature='lifespan'
			age_feature='age'			

		mut_lifespans=mut_worms.get_feature(lifespan_feature)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans/24,10)))
		mut_extended_max_age=(int)(round(numpy.percentile(mut_lifespans/24,75)))
		control_lifespans=control_worms.get_feature(lifespan_feature)
		control_max_age=(int)(round(numpy.percentile(control_lifespans/24,10)))
		control_extended_max_age=(int)(round(numpy.percentile(control_lifespans/24,75)))
		mut_ages=[]
		control_ages=[]	
		for i in range(min_age,mut_extended_max_age+1):
			mut_ages.append(i)
		for i in range(min_age,control_extended_max_age+1):
			control_ages.append(i)			
		features_towrite=['age']
		mut_correlations_towrite=[mut_ages[1::]]
		control_correlations_towrite=[control_ages[1::]]
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[1]
			save_name='/Volumes/9karray/Kinser_Holly/all_daf16_correlation_plots/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_daf16_correlation_plots/'+miRNA+'_'+key+'.svg'
			mut_correlations=[]
			control_correlations=[]
			mut_p_values=[]
			control_p_values=[]
			mut_n=[]
			control_n=[]
			try:	
				for i in range(0,len(mut_ages)-1):
					filtered_mut_worms=mut_worms.filter(lambda worm: getattr(worm,lifespan_feature) > mut_ages[i+1]*24)
					mut_results=filtered_mut_worms.regress(get_average_wrapper(mut_ages[i]*24,mut_ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(mut_ages[i]*24,mut_ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
					mut_correlations.append(mut_results.R2)
					filtered_mut_lifespans=filtered_mut_worms.get_feature(lifespan_feature)
					slope, intercept,rvalue,mut_pvalue,stderror=scipy.stats.linregress(filtered_mut_lifespans,mut_results.y_est)
					mut_p_values.append(mut_pvalue)
					mut_n.append(len(filtered_mut_worms))
				for i in range(0,len(control_ages)-1):
					filtered_control_worms=control_worms.filter(lambda worm: getattr(worm,lifespan_feature) > control_ages[i+1]*24)
					control_results=filtered_control_worms.regress(get_average_wrapper(control_ages[i]*24,control_ages[i+1]*24,key+'_z',age_feature=age_feature),get_slope_wrapper(control_ages[i]*24,control_ages[i+1]*24,key+'_z',age_feature=age_feature),target=lifespan_feature)
					control_correlations.append(control_results.R2)
					filtered_control_lifespans=filtered_control_worms.get_feature(lifespan_feature)
					slope, intercept,rvalue,control_pvalue,stderror=scipy.stats.linregress(filtered_control_lifespans,control_results.y_est)
					control_p_values.append(control_pvalue)	
					control_n.append(len(filtered_control_worms))
			except:
				continue		
			mut_correlations_towrite.append(mut_correlations)
			control_correlations_towrite.append(control_correlations)
			features_towrite.append(key)
			features_towrite.append('p_value')
			mut_correlations_towrite.append(mut_p_values)
			control_correlations_towrite.append(control_p_values)
			features_towrite.append('n')
			mut_correlations_towrite.append(mut_n)
			control_correlations_towrite.append(control_n)
			mut_p_x=[]
			mut_p_y=[]
			for i in range(0,len(mut_p_values)):
				if mut_p_values[i]<.05 and mut_correlations[i]>=.1:
					mut_p_y.append(mut_correlations[i]+.03)
					mut_p_x.append(mut_ages[1::][i])
			control_p_x=[]
			control_p_y=[]
			for i in range(0,len(control_p_values)):
				if control_p_values[i]<.05 and control_correlations[i]>=.1:
					control_p_y.append(control_correlations[i]+.03)
					control_p_x.append(control_ages[1::][i])

			orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
			plt.legend(handles=[orangered_patch,indigo_patch])
			plt.scatter(mut_p_x,mut_p_y,marker=(6,2,0),c='orangered',s=50,alpha=.7)
			plt.scatter(control_p_x,control_p_y,marker=(6,2,0),c='indigo',s=50,alpha=.7)
			plt.scatter(mut_ages[1::],mut_correlations,c='orangered',marker='o',s=50,edgecolor='orangered',alpha=.7)
			plt.scatter(control_ages[1::],control_correlations,c='indigo',marker='o',s=50,edgecolor='indigo',alpha=.7)
			plt.plot(mut_ages[1:mut_ages.index(mut_max_age)+1],mut_correlations[0:mut_ages.index(mut_max_age)], c='orangered',alpha=.7)
			plt.plot(control_ages[1:control_ages.index(control_max_age)+1],control_correlations[0:control_ages.index(control_max_age)], c='indigo',alpha=.7)
			plt.plot(mut_ages[mut_ages.index(mut_max_age)::],mut_correlations[mut_ages.index(mut_max_age)-1::],c='orangered',linestyle='--',alpha=.7)
			plt.plot(control_ages[control_ages.index(control_max_age)::],control_correlations[control_ages.index(control_max_age)-1::],c='indigo',linestyle='--',alpha=.7)	
			plt.title('P'+miRNA + ' lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
			plt.xlabel('Age (day post-hatch)', fontdict=label_fontdict, fontsize=24)
			plt.ylabel('Coefficient of determination (r2)',fontdict=label_fontdict, fontsize=24)
			plt.ylim(0,.6)
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
		n=len(mut_ages)
		rows=[[] for _ in range(n)]
		for col in mut_correlations_towrite:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_daf16_correlation_plots/'+miRNA+'_'+'mutant_correlations.tsv',[features_towrite]+rows)
		n=len(control_ages)
		rows=[[] for _ in range(n)]
		for col in control_correlations_towrite:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_daf16_correlation_plots/'+miRNA+'_'+'control_correlations.tsv',[features_towrite]+rows)