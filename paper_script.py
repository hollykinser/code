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

def percentile_wrapper(lifespans, threshold1, threshold2):
	def threshold_worms_by_lifespan(worm):
		low,high=numpy.percentile(lifespans,[threshold1, threshold2])
		return low<worm.lifespan<=high
	return threshold_worms_by_lifespan
def smooth(y, box_pts):
	box=numpy.ones(box_pts)/box_pts
	y_smooth=numpy.convolve(y,box,mode='same')
	return y_smooth


def all_wild_type_plot_consistency(root_folder):
	

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
				worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
				worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

		lifespans=worms.get_feature('lifespan')
		high_worms=worms.filter(lambda worm: worm.lifespan > numpy.percentile(lifespans,85))
		low_worms=worms.filter(lambda worm: worm.lifespan < numpy.percentile(lifespans,15))

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			averaged_worms=worm_data.meta_worms({'high_worms':high_worms,'low_worms':low_worms}, key+'_z')
			figure=averaged_worms.plot_timecourse(key+'_z',time_units='days',min_age=3*24,max_age=numpy.percentile(lifespans,10))
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_consistency/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_consistency/'+miRNA+'_'+key+'.svg'
			plt.plot(high_worms[0].td.age[(high_worms[0].td.age>3*24)&(high_worms[0].td.age<numpy.percentile(lifespans,10))]/24,numpy.zeros(len(high_worms[0].td.age[(high_worms[0].td.age>3*24)&(high_worms[0].td.age<numpy.percentile(lifespans,10))])),color='gray',linestyle='--')

			
			#for worm in high_worms:
				#plt.plot(worm.td.age[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))]/24,smooth(getattr(worm.td,key+'_z')[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))],4), color='goldenrod',alpha=.5)
	  	
			#for worm in low_worms:
				#plt.plot(worm.td.age[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))]/24,smooth(getattr(worm.td,key+'_z')[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))],4), color='indigo',alpha=.5)
			plt.xlabel('Time (days)',fontdict=label_fontdict)
			plt.ylabel('Z score ('+gfp_measures[key]+')',fontdict=label_fontdict)
			plt.title('P'+miRNA +'::GFP expression consistency over time',fontdict=title_fontdict)
			ymin, ymax=plt.ylim()
			plt.ylim(ymin=-3,ymax=3)	
			text='n = ' + (str)(len(high_worms))+ '    '+(str)(len(low_worms))
			plt.figtext(.5, .15, text, fontdict=text_fontdict)
			plt.margins(.1,.1)
			print("Saving "+ save_name)
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()

def all_wild_type_lifespan_hist(root_folder,overwrite=True):
	
	#plots histogram of lifespans with kde distribution
	
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


def all_wild_type_population_trajectory(root_folder,overwrite=True,min_age=24,time_units='days'):
	
	#plots population-level expression trace

	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		#looks for manually-set parameters in 'parameters.tsv' file (useful for strains where it's known that expression is not detectable at later timepoints)
		
		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
		else:
			max_age=numpy.inf		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
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
						median_gfp[exp] = numpy.median([d[:,1].median() for d in data])
				
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

			else:
				for worm in worms:
					worm.td.scaled_gfp = getattr(worm.td,key)

			lifespans=worms.get_feature('lifespan')
			bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}



			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
			figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='age',min_age=min_age,max_age=max_age,time_units=time_units)
			
			cohorts=[]

			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	

			text='n = ' + (str)(cohorts[0])
			plt.figtext(.8, .3, text, fontdict=text_fontdict, fontsize=24)
			plt.title('P'+miRNA+'::GFP expression',fontdict=title_fontdict, loc='left')
			ax = plt.gca() 
			ax.set_xlabel('Days', fontdict=label_fontdict,fontsizer=24)
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
def all_wild_type_population_trajectory_with_stddev(root_folder,overwrite=True,min_age=24,time_units='hours'):
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
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
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
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

			else:
				for worm in worms:

						worm.td.scaled_gfp = getattr(worm.td,key)

			lifespans=worms.get_feature('lifespan')
			bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}



			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
			
			cohorts=[]

			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			trend_x,mean_trend,std_trend=worms.z_transform('scaled_gfp',min_age=72,max_age=max_age)
			plt.plot(trend_x/24,mean_trend,c='mediumspringgreen')
			plt.plot(trend_x/24,mean_trend+std_trend,c='mediumspringgreen',alpha=.5,linestyle='--')
			plt.plot(trend_x/24,mean_trend-std_trend,c='mediumspringgreen',alpha=.5,linestyle='--')
			text='n = ' + (str)(cohorts)
			plt.figtext(.5, .15, text, fontdict=text_fontdict,fontsize=28)
			plt.title('Average P' + miRNA+ '::GFP expression',fontdict=title_fontdict)
			plt.ylabel('P'+miRNA+'::GFP expression ('+value+')',fontdict=label_fontdict,fontsize=24)
			plt.xlabel('Days',fontdict=label_fontdict,fontsize=24)
			ax = plt.gca() 
			ax.set_xlim(left=-.5,right=19)
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
def all_wild_type_population_trajectory_with_stddev_individual(root_folder,overwrite=True,min_age=24,time_units='hours'):
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		
		for exp_dir in exp_dirs:
			worms=process_worms([exp_dir], prefixes=[exp_dir])

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24
			else:
				max_age=numpy.inf		

			for key,value in gfp_measures.items():
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory_with_stddev_individual/'+miRNA+'_'+exp_dir[-9:-1]+'_'+key+'.svg'

				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue
				try:
					worms.get_time_range(key,24,numpy.inf)
				except:
					continue	
			
			

				lifespans=worms.get_feature('lifespan')
				bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}



				averaged_worms = worm_data.meta_worms(bins, key, age_feature='age')
			
				cohorts=[]

				for mew, mewmew in bins.items():
					cohorts.append(len(mewmew))	
				trend_x,mean_trend,std_trend=worms.z_transform(key,min_age=24,max_age=max_age)
				plt.plot(trend_x,mean_trend,c='mediumspringgreen')
				plt.plot(trend_x,mean_trend+std_trend,c='mediumspringgreen',alpha=.5,linestyle='--')
				plt.plot(trend_x,mean_trend-std_trend,c='mediumspringgreen',alpha=.5,linestyle='--')
				text='n = ' + (str)(cohorts)
				plt.figtext(.5, .15, text, fontdict=text_fontdict,fontsize=24)
				plt.title('Average ' + miRNA+ '::GFP expression profiles',fontdict=title_fontdict)
				plt.ylabel('P'+miRNA+'::GFP expression ('+value+')',fontdict=label_fontdict)
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.gcf().clf
				plt.close()			
def all_mutant_population_trajectory(root_folder,overwrite=True,min_age=24,time_units='hours'):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])

		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
		else:
			max_age=numpy.inf		

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			try:
				mut_worms.get_time_range(key,24,numpy.inf)
			except:
				continue	
			
			if len(mut_exp_dirs)>1:
				mut_median_gfp={}
				control_median_gfp={}
				mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
				control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
	
				for exp, wormies in mut_groups.items():
						mut_data = wormies.get_time_range(key,min_age,max_age)
						mut_median_gfp[exp] = numpy.median([d[:,1].mean() for d in mut_data])
				for exp, wormies in control_groups.items():
						control_data = wormies.get_time_range(key,min_age,max_age)			
						control_median_gfp[exp] = numpy.median([d[:,1].mean() for d in control_data])
				for worm in mut_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ mut_median_gfp[exp] * mut_median_gfp[mut_exp_dirs[0]]
				for worm in control_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ control_median_gfp[exp] * control_median_gfp[control_exp_dirs[0]]
	
						

			else:
				for worm in control_worms:
					worm.td.scaled_gfp = getattr(worm.td,key)
				for worm in mut_worms:
					worm.td.scaled_gfp = getattr(worm.td,key)		

			mut_lifespans=mut_worms.get_feature('lifespan')
			control_lifespans=control_worms.get_feature('lifespan')
			mut_bins={'['+(str)(mut_lifespans.min())+'-'+(str)(mut_lifespans.max())+']':mut_worms}
			control_bins={'['+(str)(control_lifespans.min())+'-'+(str)(control_lifespans.max())+']':control_worms}


			mut_averaged_worms = worm_data.meta_worms(mut_bins, 'scaled_gfp', age_feature='age')
			control_averaged_worms = worm_data.meta_worms(control_bins, 'scaled_gfp', age_feature='age')
			
			mut_cohorts=[]
			control_cohorts=[]

			for mew, mewmew in mut_bins.items():
				mut_cohorts.append(len(mewmew))	
			for mew, mewmew in control_bins.items():
				control_cohorts.append(len(mewmew))		
			mut_time_ranges=mut_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,'age')
			control_time_ranges=control_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,'age')
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
			plt.figtext(.5, .15, mut_text, fontdict=text_fontdict)
			plt.figtext(.5, .2, control_text, fontdict=text_fontdict)
			plt.title('Average ' +'P'+miRNA+'::GFP expression profiles',fontdict=title_fontdict)
			plt.ylabel('P'+miRNA+'::GFP expression ('+value+')',fontdict=label_fontdict)
			plt.margins(.15,.15)
			save_name='/Volumes/9karray/Kinser_Holly/all_mutant_population_trajectory/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_mutant_population_trajectory/'+miRNA+'_'+key+'.svg'
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.clf()
			plt.close()
def all_daf16_population_trajectory(root_folder,overwrite=True,min_age=24,time_units='hours'):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])

		if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
			parameters=list(parameters)
			max_age=parameters[header.index('max_age')]
			max_age=((int)(max_age[0]))*24
		else:
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
				mut_groups=mut_worms.group_by([w.name.split()[0] for w in mut_worms])
				control_groups=control_worms.group_by([w.name.split()[0] for w in control_worms])		
	
				for exp, wormies in mut_groups.items():
						mut_data = wormies.get_time_range(key,min_age,max_age)
						mut_median_gfp[exp] = numpy.median([d[:,1].mean() for d in mut_data])
				for exp, wormies in control_groups.items():
						control_data = wormies.get_time_range(key,min_age,max_age)			
						control_median_gfp[exp] = numpy.median([d[:,1].mean() for d in control_data])
				for worm in mut_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ mut_median_gfp[exp] * mut_median_gfp[mut_exp_dirs[0]]
				for worm in control_worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ control_median_gfp[exp] * control_median_gfp[control_exp_dirs[0]]
	
						

			else:
				for worm in control_worms:
					worm.td.scaled_gfp = getattr(worm.td,key)
				for worm in mut_worms:
					worm.td.scaled_gfp = getattr(worm.td,key)		

			mut_lifespans=mut_worms.get_feature('lifespan')
			control_lifespans=control_worms.get_feature('lifespan')
			mut_bins={'['+(str)(mut_lifespans.min())+'-'+(str)(mut_lifespans.max())+']':mut_worms}
			control_bins={'['+(str)(control_lifespans.min())+'-'+(str)(control_lifespans.max())+']':control_worms}


			mut_averaged_worms = worm_data.meta_worms(mut_bins, 'scaled_gfp', age_feature='age')
			control_averaged_worms = worm_data.meta_worms(control_bins, 'scaled_gfp', age_feature='age')
			
			mut_cohorts=[]
			control_cohorts=[]

			for mew, mewmew in mut_bins.items():
				mut_cohorts.append(len(mewmew))	
			for mew, mewmew in control_bins.items():
				control_cohorts.append(len(mewmew))		
			mut_time_ranges=mut_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,'age')
			control_time_ranges=control_averaged_worms.get_time_range('scaled_gfp',min_age,max_age,'age')
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
def all_keypoint_population_trajectory(root_folder,overwrite=True,min_age=48,time_units='hours'):
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

		for key in list(worms[0].td._keys()):
			if 'age' not in key and 'timepoint' not in key and 'timestamp' not in key and 'z' not in key and 'centroid_dist' not in key:
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_population_trajectory/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_population_trajectory/'+miRNA+'_'+key+'.svg'
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue
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
							worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						
						

				else:
					for worm in worms:

						worm.td.scaled_gfp = getattr(worm.td,key)

				lifespans=worms.get_feature('lifespan')
				bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}



				averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
				figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='age',min_age=min_age,max_age=max_age,time_units='days')
			
				cohorts=[]

				for mew, mewmew in bins.items():
					cohorts.append(len(mewmew))	
			
				text='n = ' + (str)(cohorts[0])
				plt.figtext(.8, .3, text, fontdict=text_fontdict)
				plt.title('P'+miRNA+'::GFP expression',fontdict=title_fontdict, loc='left')
				ax = plt.gca() 
				ax.set_xlabel('Days', fontdict=label_fontdict, fontsize=24)
				ax.xaxis.set_label_coords(.95, .05)
				ax.set_ylabel('Expression ('+key+')',fontdict=label_fontdict, fontsize=24)
				ax.set_xlim(left=-.5,right=19)
				ax.yaxis.set_label_coords(.05,.5)
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()
def all_keypoint_population_trajectory_cohorts(root_folder,overwrite=True,min_age=48,time_units='hours'):
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

		for key in list(worms[0].td._keys()):
			if 'age' not in key and 'timepoint' not in key and 'timestamp' not in key and 'z' not in key and 'centroid_dist' not in key:
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue
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
							worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
							worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

				else:
					for worm in worms:

							worm.td.scaled_gfp = getattr(worm.td,key)

				lifespans=worms.get_feature('lifespan')
				bins=worms.bin('lifespan',nbins=5, equal_count=False)



				averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
				figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='age',min_age=min_age,max_age=max_age,time_units='days')
			
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
				plt.ylabel('P'+miRNA+'::GFP expression ('+key+')',fontdict=label_fontdict, fontsize=24)
				plt.margins(.15,.15)
				print('Saving '+save_name)	
				plt.savefig(save_name)
				plt.savefig(save_name_box)
				plt.gcf().clf
				plt.close()						
def all_nonmir_population_trajectory(root_folder,overwrite=True,min_age=24,time_units='hours'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_population_trajectory/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_population_trajectory/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
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
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

			else:
				for worm in worms:

						worm.td.scaled_gfp = getattr(worm.td,key)

			lifespans=worms.get_feature('lifespan')
			bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}



			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
			figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='age',min_age=min_age,max_age=max_age,time_units='days')
			
			cohorts=[]

			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			
			text='n = ' + (str)(cohorts)
			plt.figtext(.5, .15, text, fontdict=text_fontdict)
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
def all_nonmir_population_trajectory_cohorts(root_folder,overwrite=True,min_age=24,time_units='hours'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			try:
				worms.get_time_range(key,min_age,numpy.inf)
			except:
				continue	
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,numpy.inf)	
						median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
						median_gfp[exp] = numpy.median([d[:,1].mean() for d in data])
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

			else:
				for worm in worms:

						worm.td.scaled_gfp = getattr(worm.td,key)

			lifespans=worms.get_feature('lifespan')
			bins=worms.bin('lifespan',nbins=5, equal_count=False)



			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
			figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='age',min_age=min_age, max_age=max_age,time_units='days')
			
			cohorts=[]

			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			
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
#generates cohort traces of average expression for each measure in gfp_measures
def all_wild_type_population_trajectory_cohorts(root_folder,overwrite=True,min_age=24,time_units='hours'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory_cohorts/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			try:
				worms.get_time_range(key,min_age,numpy.inf)
			except:
				continue	
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,numpy.inf)	
						median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
						median_gfp[exp] = numpy.median([d[:,1].mean() for d in data])
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

			else:
				for worm in worms:

						worm.td.scaled_gfp = getattr(worm.td,key)

			lifespans=worms.get_feature('lifespan')
			bins=worms.bin('lifespan',nbins=5, equal_count=False)



			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='age')
			figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='age',min_age=min_age, max_age=max_age,time_units='days')
			
			cohorts=[]

			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			
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
def all_wild_type_population_trajectory_cohorts_ghost(root_folder,overwrite=True,min_age=24,time_units='days'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_population_trajectory_cohorts_ghost/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_population_trajectory_cohorts_ghost/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			try:
				worms.get_time_range(key,min_age,numpy.inf)
			except:
				continue	
			
			if len(exp_dirs)>1:
				median_gfp={}
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
						data = wormies.get_time_range(key,min_age,numpy.inf)	
						median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))	
						median_gfp[exp] = numpy.median([d[:,1].mean() for d in data])
				for worm in worms:
						exp = worm.name.split()[0]
						worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[exp_dirs[0]]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
						

			else:
				for worm in worms:

						worm.td.scaled_gfp = getattr(worm.td,key)

			lifespans=worms.get_feature('lifespan')
			bins=worms.bin('lifespan',nbins=5, equal_count=True)



			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature='ghost_age')
			figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature='ghost_age',time_units=time_units)
			
			cohorts=[]

			for mew, mewmew in bins.items():
				cohorts.append(len(mewmew))	
			
			text='n = ' + (str)(cohorts)
			ax = plt.gca() 
			ax.set_xlabel('Days', fontdict=label_fontdict, fontsize=24)
			ax.xaxis.set_label_coords(.95, .05)
			ax.set_ylabel('Expression ('+value+')',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			#ax.set_xlim(left=-.5,right=19)
			plt.figtext(.5, .15, text, fontdict=text_fontdict, fontsize=28)
			plt.title('P'+miRNA+'::GFP',fontdict=title_fontdict, loc='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()			
def all_nonmir_regression_on_slope_and_mean(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			lifespans=worms.get_feature(target)
			max_age=round(numpy.percentile(lifespans,10))
		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
		
			try:
				results=worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
			except:	
				continue	
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)
			ptext="p = " + (str)(round(p_value, 3))
			plt.figtext(.15,.75,ptext, fontsize=20,ha='left')	
			color_vals = colorize.scale(lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter(lifespans,results.y_est,c=colors)
			(m,b) = numpy.polyfit(lifespans, results.y_est, 1)
			yp=numpy.polyval([m,b], lifespans)
			mew, mewmew=zip(*sorted(zip(lifespans, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="$r^{2}$ = "+(str)(round(results.R2,3))
			more_text="n= " + (str)(len(lifespans))
			plt.title('Regression on Slope and Mean ' +miRNA+'::GFP expression profiles '+(str)(min_age) + ' to '+(str)(max_age)+ ' '+key,fontdict=title_fontdict)
			plt.xlabel('Actual lifespan (days)',fontdict=label_fontdict)
			plt.ylabel("Predicted lifespan (days)",fontdict=label_fontdict)
			plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()

#regresses on slope and mean for each measure in gfp_measures, from 3 dph to timepoint at which 90% of population is alive
def all_wild_type_regression_on_slope_and_mean_same_x(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(target)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(target)
			#filtered_lifespans=[i for i in lifespans if i>max_age]
			
			try:
				filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
				filtered_lifespans=filtered_worms.get_feature('lifespan')
	
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
				slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
				average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),target=target)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
			except:	
				continue

			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)		
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			features_to_write.append(key+'_p_value')
			correlations_to_write.append(p_value)
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
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=100/24, right=450/24)
			#ax.set_ylim(bottom=150,top=400)

			#plt.xlabel('Actual lifespan (hours)',fontdict=label_fontdict)
			#plt.ylabel("Predicted lifespan (hours)",fontdict=label_fontdict)
			#plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			#plt.figtext(.15,.75,ptext, fontsize=20,ha='left')
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
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
		datafile.write_delimited(subdirectory+'/regression_correlations.tsv',[features_to_write]+[correlations_to_write])
def all_nonmir_regression_on_slope_and_mean_same_x(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_regression_on_slope_and_mean_same_x/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(target)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(target)
			#filtered_lifespans=[i for i in lifespans if i>max_age]
			
			try:
				filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
				filtered_lifespans=filtered_worms.get_feature('lifespan')
	
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
				slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
				average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),target=target)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
			except:	
				continue

			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)		
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			features_to_write.append(key+'_p_value')
			correlations_to_write.append(p_value)
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
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (days)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (days)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			ax.set_xlim(left=100/24, right=450/24)
			#ax.set_ylim(bottom=150,top=400)

			#plt.xlabel('Actual lifespan (hours)',fontdict=label_fontdict)
			#plt.ylabel("Predicted lifespan (hours)",fontdict=label_fontdict)
			#plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			#plt.figtext(.15,.75,ptext, fontsize=20,ha='left')
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
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
		datafile.write_delimited(subdirectory+'/regression_correlations.tsv',[features_to_write]+[correlations_to_write])		
def all_wild_type_regression_on_slope_and_mean(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(target)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(target)
			#filtered_lifespans=[i for i in lifespans if i>max_age]
			
			try:
				filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
				filtered_lifespans=filtered_worms.get_feature('lifespan')
	
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
				slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
				average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),target=target)
				correlations_to_write.append(slope_results.R2)
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(slope_results.X,filtered_lifespans)
				correlations_to_write.append(r_value)
				correlations_to_write.append(p_value)
				correlations_to_write.append(average_results.R2)
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(average_results.X,filtered_lifespans)
				correlations_to_write.append(r_value)
				correlations_to_write.append(p_value)
				correlations_to_write.append(results.R2)
		
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)
				correlations_to_write.append(p_value)	
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_slope_direction')
				features_to_write.append(key+'_slope_p_value')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_mean_direction')
				features_to_write.append(key+'mean_p_value')
				features_to_write.append(key+'_joint')
				features_to_write.append(key+'_joint_p_value')
			except:	
				continue

			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)		
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter(filtered_lifespans,results.y_est,c=colors)
			(m,b) = numpy.polyfit(filtered_lifespans, results.y_est, 1)
			yp=numpy.polyval([m,b], filtered_lifespans)
			mew, mewmew=zip(*sorted(zip(filtered_lifespans, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="R2 = "+(str)(round(results.R2,3))
			ptext="p = " + (str)(round(p_value, 3))
			more_text="n= " + (str)(len(filtered_lifespans))
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext,fontdict=title_fontdict)
			ax = plt.gca() 
			ax.set_xlabel('Lifespan (hours)', fontdict=label_fontdict,fontsize=24)
			ax.xaxis.set_label_coords(.85, .05)
			ax.set_ylabel('Expression-predicted lifespan (hours)',fontdict=label_fontdict, fontsize=24)
			ax.yaxis.set_label_coords(.05,.5)
			plt.figtext(.15, .15, more_text, fontsize=28, ha='left')
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
		datafile.write_delimited(subdirectory+'/regression_correlations.tsv',[features_to_write]+[correlations_to_write])
def all_wild_type_regression_on_slope_and_mean_sliding(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(target)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(target)
			#filtered_lifespans=[i for i in lifespans if i>max_age]
			ages=[]
			correlations=[]
			windows=[]
			for i in range(min_age,(int)(max_age)+(int)(max_age)%12,12):
				ages.append(i)
			
			for i in range(0, len(ages)-1):
				left_window=ages[i]
				for j in range(ages.index(left_window)+1, len(ages)):
					right_window=ages[j]
					filtered_worms=worms.filter(lambda worm: worm.lifespan > right_window)
					filtered_lifespans=filtered_worms.get_feature('lifespan')
					try:
						results=filtered_worms.regress(get_average_wrapper(left_window,right_window,key+'_z'),get_slope_wrapper(left_window,right_window,key+'_z'),target=target)
					except:
						continue
					correlations.append(results.R2)
					windows.append([left_window,right_window])
				
			best_window=windows[correlations.index(max(correlations))]
			results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z'),get_slope_wrapper(best_window[0],best_window[1],key+'_z'),target=target)		
			slope_results=filtered_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key+'_z'),target=target)
			average_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z'),target=target)
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
			ptext="p = " + (str)(round(p_value, 3))
			more_text="n= " + (str)(len(results.resid))
			plt.title('P'+miRNA+'::GFP expression vs. lifespan: '+ftext + (str)(best_window),fontdict=title_fontdict)
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
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_sliding/'+miRNA+'_'+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])
			
def all_wild_type_regression_on_slope_and_mean_individual(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:
		features_to_write=[]
		correlations_to_write=[]

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])

		for key,value in gfp_measures.items():
			for exp_dir in exp_dirs:
				worms=process_worms([exp_dir], prefixes=[' '])
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_individual/'+miRNA+'_'+exp_dir[-9::]+'_'+key+'.svg'
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue	
			
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
	
					results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
					slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age,key+'_z'),target=target)
					average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),target=target)
					correlations_to_write.append(slope_results.R2)
					correlations_to_write.append(average_results.R2)
					correlations_to_write.append(results.R2)
					features_to_write.append(key+'_slope')
					features_to_write.append(key+'_mean')
					features_to_write.append(key+'_joint')
				except:	
					continue

				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)		
				color_vals = colorize.scale(filtered_lifespans, output_max=1)
				colors = colorize.color_map(color_vals, uint8=False)	
				plt.scatter(filtered_lifespans,results.y_est,c=colors)
				(m,b) = numpy.polyfit(filtered_lifespans, results.y_est, 1)
				yp=numpy.polyval([m,b], filtered_lifespans)
				mew, mewmew=zip(*sorted(zip(filtered_lifespans, yp)))
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
				plt.savefig
				plt.gcf().clf
				plt.close()

def all_wild_type_regression_on_slope_and_mean_plus_autofluorescence_plus_length(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence_plus_length/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence_plus_length/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

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
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),get_average_wrapper(min_age,max_age,'length'),get_slope_wrapper(min_age,max_age,'length'),target=target)
				gfp_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'))
				auto_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'))
				gfp_plus_auto_results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'))
				correlations_to_write.append(gfp_results.R2)
				correlations_to_write.append(auto_results.R2)
				correlations_to_write.append(gfp_plus_auto_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_gfp')
				features_to_write.append('auto')
				features_to_write.append(key+'_gfp_plus_auto')
				features_to_write.append(key+'_gfp_plus_auto_plus_length')

			except:	
				continue	
			filtered_lifespans=filtered_worms.get_feature('lifespan')	
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)
			ptext="p = " + (str)(round(p_value, 3))
			plt.figtext(.15,.75,ptext, fontsize=20,ha='left')		
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter(filtered_lifespans,results.y_est,c=colors)
			(m,b) = numpy.polyfit(filtered_lifespans, results.y_est, 1)
			yp=numpy.polyval([m,b], filtered_lifespans)
			mew, mewmew=zip(*sorted(zip(filtered_lifespans, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="$r^{2}$ = "+(str)(round(results.R2,3))
			more_text="n= " + (str)(len(filtered_lifespans))
			plt.title('Regression on Slope and Mean ' +miRNA+'::GFP expression profiles '+(str)(min_age) + ' to '+(str)(max_age)+ ' '+key,fontdict=title_fontdict)
			plt.xlabel('Actual lifespan (hours)',fontdict=label_fontdict)
			plt.ylabel("Predicted lifespan (hours)",fontdict=label_fontdict)
			plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
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
		datafile.write_delimited(subdirectory+'/auto_length_regression_correlations.tsv',[features_to_write]+[correlations_to_write])			

def all_wild_type_regression_on_slope_and_mean_plus_autofluorescence(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_plus_autofluorescence/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(target)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(target)
			filtered_lifespans=[i for i in lifespans if i>max_age]
		
			try:
				filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),target=target)
			except:	
				continue

			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)
			ptext="p = " + (str)(round(p_value, 3))
			plt.figtext(.15,.75,ptext, fontsize=20,ha='left')			
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter(filtered_lifespans,results.y_est,c=colors)
			(m,b) = numpy.polyfit(filtered_lifespans, results.y_est, 1)
			yp=numpy.polyval([m,b], filtered_lifespans)
			mew, mewmew=zip(*sorted(zip(filtered_lifespans, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="$r^{2}$ = "+(str)(round(results.R2,3))
			more_text="n= " + (str)(len(filtered_lifespans))
			plt.title('Regression on Slope and Mean ' +miRNA+'::GFP expression profiles '+(str)(min_age) + ' to '+(str)(max_age)+ ' '+key,fontdict=title_fontdict)
			plt.xlabel('Actual lifespan (hours)',fontdict=label_fontdict)
			plt.ylabel("Predicted lifespan (hours)",fontdict=label_fontdict)
			plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()			
#regresses on slope and mean for each measure in gfp_measures, from 3 dph to timepoint at which 90% of population is alive. 
#Controls for autofluorescence in regression which eliminates spurious correlation with dim reporters			
def all_wild_type_regression_on_slope_and_mean_controlled_for_autofluorescence(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=sorted([subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()])
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])

		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_regression_on_slope_and_mean_controlled_for_autofluorescence/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_regression_on_slope_and_mean_controlled_for_autofluorescence/'+miRNA+'_'+key+'.svg'
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue	
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			if pathlib.Path(subdirectory+'/'+'parameters.tsv').exists():
				header,parameters=datafile.read_delimited(subdirectory+'/'+'parameters.tsv')
				parameters=list(parameters)
				max_age=parameters[header.index('max_age')]
				max_age=((int)(max_age[0]))*24			
			else:
				lifespans=worms.get_feature(target)
				max_age=round(numpy.percentile(lifespans,10))	
			
			lifespans=worms.get_feature(target)
			filtered_lifespans=[i for i in lifespans if i>max_age]
		
		
			filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
			results=filtered_worms.regress(get_average_wrapper(min_age,max_age,key+'_z'),get_slope_wrapper(min_age,max_age,key+'_z'),target='lifespan',control_features=[get_average_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z'),get_slope_wrapper(min_age,max_age,'green_yellow_excitation_autofluorescence_percentile_95_z')])
		
			slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(filtered_lifespans,results.y_est)
			ptext="p = " + (str)(round(p_value, 3))
			plt.figtext(.15,.75,ptext, fontsize=20,ha='left')		
			color_vals = colorize.scale(filtered_lifespans, output_max=1)
			colors = colorize.color_map(color_vals, uint8=False)	
			plt.scatter(filtered_lifespans,results.y_est,c=colors)
			(m,b) = numpy.polyfit(filtered_lifespans, results.y_est, 1)
			yp=numpy.polyval([m,b], filtered_lifespans)
			mew, mewmew=zip(*sorted(zip(filtered_lifespans, yp)))
			mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
			plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
			ftext="$r^{2}$ = "+(str)(round(results.R2,3))
			more_text="n= " + (str)(len(filtered_lifespans))
			plt.title('Regression on Slope and Mean ' +miRNA+'::GFP expression profiles '+(str)(min_age) + ' to '+(str)(max_age)+ ' '+key,fontdict=title_fontdict)
			plt.xlabel('Actual lifespan (hours)',fontdict=label_fontdict)
			plt.ylabel("Predicted lifespan (hours)",fontdict=label_fontdict)
			plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
			plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
			plt.margins(.15,.15)
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()


def all_straightened_worm_images(path_to_parameters):

	header,parameters=datafile.read_delimited(path_to_parameters+'/image_parameters.tsv')
	parameters=list(parameters)[0]
	maximum=parameters[header.index('maximum')]
	maximum=((int)(maximum))
	minimum=parameters[header.index('minimum')]
	minimum=((int)(minimum))
	gamma=parameters[header.index('gamma')]
	gamma=((int)(gamma))
	exp_root=parameters[header.index('exp_root')]
	position_root=parameters[header.index('position_root')]
	labels=position_root.split('/')
	annotations=load_data.read_annotations(exp_root)[labels[-1]]
	p=pathlib.Path(position_root)
	gfp=sorted(p.glob('* gfp.png'))
	bf=sorted(p.glob('* bf.png'))
	gfp_images=[freeimage.read(path) for path in gfp]
	bf_images=[freeimage.read(path) for path in bf]
	position_info,timepoint_info=annotations
	timepoints=timepoint_info.keys()
	width_tcks=[timepoint_info[key]['pose'][1] for key in timepoints]
	center_tcks=[timepoint_info[key]['pose'][0] for key in timepoints]
	ages=[timepoint_info[key]['age'] for key in timepoints]
	ages=[round(age/24,2)for age in ages]
	bf_splines=[]
	gfp_splines=[]


	for i in range(0, len(bf_images)):
		bf_spline=worm_spline.to_worm_frame(bf_images[i],center_tcks[i],width_tcks[i],width_margin=0,standard_length=1000)
		gfp_spline=worm_spline.to_worm_frame(gfp_images[i],center_tcks[i],width_tcks[i],width_margin=0,standard_length=1000)
		bf_splines.append(bf_spline)
		gfp_splines.append(gfp_spline)
	for bf_spline,gfp_spline,width_tck,age in zip(bf_splines,gfp_splines,width_tcks,ages):
		save_dir='/Volumes/9karray/Kinser_Holly/all_straightened_worm_images/'+labels[3]+'/'+labels[4]
		save_name_box='/Users/pincuslab/Box/miRNA Data/all_straightened_worm_images/'+labels[3]+'/'+labels[4]
		if pathlib.Path(save_dir).exists()==False:
			os.makedirs(save_dir)
		mask=worm_spline.worm_frame_mask(width_tck,gfp_spline.shape,antialias=True)
		
		gfp_scaled=colorize.scale(gfp_spline,minimum,maximum,gamma=gamma).astype('uint8')
		bf_scaled=colorize.scale(bf_spline,1000,20000,gamma=1).astype('uint8')
	
		
		freeimage.write(numpy.dstack((bf_scaled,bf_scaled,bf_scaled,mask)),save_dir+'/'+(str)(age) +'_bf.png')
		freeimage.write(numpy.dstack((gfp_scaled,gfp_scaled,gfp_scaled,mask)),save_dir+'/'+(str)(age) +'_gfp.png')

def all_keypoint_regression_on_slope_and_mean_controlled(root_folder,overwrite=True,min_age=3,target='lifespan',subdirectories=None):
	if subdirectories==None:
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
		else:
			if len(exp_dirs)>1:
				print(True)
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
			#max_age=(int)(round((numpy.percentile(possible_max_ages,90))))
			max_age=11
			print(max_age)
			#if max_age>numpy.percentile(lifespans/24,20):
				#print('True')
				#max_age=round(numpy.percentile(lifespans/24,20))	
			#print(max_age)
		

		
		keypoint_keys=list(worms[0].td._keys())
		for key in keypoint_keys:
			if 'z' in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key and 'surface_area' and 'max_width' not in key and 'volume' not in key and 'rms_dist' not in key and 'projected_area' not in key and 'length' not in key and 'centroid_dist' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				
				for i in range(0, len(keypoint_keys)):
					if 'z' in key and 'age' not in keypoint_keys[i] and 'stage' not in keypoint_keys[i] and 'timepoint' not in keypoint_keys[i] and 'timestamp' not in keypoint_keys[i] and 'surface_area' and 'max_width' not in keypoint_keys[i] and 'volume' not in keypoint_keys[i] and 'rms_dist' not in keypoint_keys[i] and 'projected_area' not in keypoint_keys[i] and 'length' not in keypoint_keys[i] and 'centroid_dist' not in keypoint_keys[i]:
						save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_controlled/'+miRNA+'_'+key+'controlled_for_'+keypoint_keys[i]+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean_controlled/'+miRNA+'_'+key+'controlled_for_'+keypoint_keys[i]+'.svg'

						filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age*24)
						uncontrolled_results=filtered_worms.regress(get_average_wrapper(min_age*24,max_age*24,key),get_slope_wrapper(min_age*24,max_age*24,key),target=target)
						results=filtered_worms.regress(get_average_wrapper(min_age*24,max_age*24,key),get_slope_wrapper(min_age*24,max_age*24,key),target=target,control_features=[get_average_wrapper(min_age*24,max_age*24,keypoint_keys[i]),get_slope_wrapper(min_age*24,max_age*24,keypoint_keys[i])])
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
						ftext='R2 = '+(str)(round(results.R2,3))
						gtext='R2 = '+(str)(round(uncontrolled_results.R2,3))
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
def all_keypoint_regression_on_slope_and_mean_plus_keypoint(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age>numpy.percentile(lifespans,20):
				max_age=round(numpy.percentile(lifespans,20))	
			print(max_age)
		

		ages=[]
		
		for i in range(min_age,max_age+1):
			ages.append(i)	
		print(ages)
		keypoint_keys=list(worms[0].td._keys())
		for key in keypoint_keys:
			if 'z' in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
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

						filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age*24)
						uncontrolled_results=filtered_worms.regress(get_average_wrapper(min_age,max_age*24,key),get_slope_wrapper(min_age,max_age*24,key),get_average_wrapper(min_age,max_age*24,keypoint_keys[i]),get_slope_wrapper(min_age,max_age*24,keypoint_keys[i]),target=target)
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

def all_keypoint_regression_on_slope_and_mean(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
				max_age=(int)(round((numpy.percentile(possible_max_ages,75))))
				if max_age>numpy.percentile(lifespans,90):
					max_age=round(max_age=numpy.percentile(lifespans,90))		
			print(max_age)
		

		ages=[]
		
		for i in range(min_age,max_age+1):
			ages.append(i)	
		print(ages)
		for key in list(worms[0].td._keys()):
			if 'z' not in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue
		
				filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age*24)
				results=filtered_worms.regress(get_average_wrapper(min_age,max_age*24,key+'_z'),get_slope_wrapper(min_age,max_age*24,key+'_z'),target=target)
				slope_results=filtered_worms.regress(get_slope_wrapper(min_age,max_age*24,key+'_z'),target=target)
				average_results=filtered_worms.regress(get_average_wrapper(min_age,max_age*24,key+'_z'),target=target)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
				lifespans=results.y_est+results.resid	
				
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(lifespans,results.y_est)
				ptext="p = " + (str)(round(p_value, 3))
				features_to_write.append('p_value')
				correlations_to_write.append(p_value)	
				#plt.figtext(.15,.75,ptext, fontsize=20,ha='left')		
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
		datafile.write_delimited(subdirectory+'/keypoint_regression_correlations.tsv',[features_to_write]+[correlations_to_write])
def all_keypoint_regression_on_slope_and_mean_sliding_controlled(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
				max_age=(int)(round((numpy.percentile(possible_max_ages,20))))
				if max_age>numpy.percentile(lifespans,20):
					max_age=round(max_age=numpy.percentile(lifespans,20))		
			print(max_age)
		
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
			if len(miRNA)>1:
				miRNA=miRNA[0]+'_'+miRNA[1]
			else:
				miRNA=miRNA[0]
		keypoint_keys=list(worms[10].td._keys())		
		for key in keypoint_keys:
			if 'z' not in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
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
						filtered_worms=worms.filter(lambda worm: worm.lifespan > right_window)
						filtered_lifespans=filtered_worms.get_feature('lifespan')
						try:	
							uncontrolled_results=filtered_worms.regress(get_average_wrapper(left_window,right_window,key+'_z'),get_slope_wrapper(left_window,right_window,key+'_z'),target=target)
						except:
							continue
						correlations.append(uncontrolled_results.R2)
						windows.append([left_window,right_window])
				
				best_window=windows[correlations.index(max(correlations))]
				print(best_window)
				for k in range(0, len(keypoint_keys)):
					if 'z' not in keypoint_keys[k] and 'age' not in keypoint_keys[k] and 'stage' not in keypoint_keys[k] and 'timepoint' not in keypoint_keys[k] and 'timestamp' not in keypoint_keys[k]:
						uncontrolled_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z'),get_slope_wrapper(best_window[0],best_window[1],key+'_z'),target=target)		
						results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z'),get_slope_wrapper(best_window[0],best_window[1],key+'_z'),target=target,control_features=[get_average_wrapper(best_window[0],best_window[1],keypoint_keys[k]),get_slope_wrapper(best_window[0],best_window[1],keypoint_keys[k])])
						lifespans=results.y_est+results.resid		
						save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_sliding_controlled/'+miRNA+'_'+key+'_controlledfor_'+keypoint_keys[k]+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean_sliding_controlled/'+miRNA+'_'+key+'_controlledfor_'+keypoint_keys[k]+'.svg'

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
						ftext='R2 = '+(str)(round(results.R2,3))
						gtext='R2 = '+(str)(round(uncontrolled_results.R2,3))
						more_text="n= " + (str)(len(lifespans))
						plt.title('P'+miRNA+'::GFP vs. lifespan: '+ftext,fontdict=title_fontdict)
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
						plt.gcf().clf
						plt.close()
def all_keypoint_regression_on_slope_and_mean_sliding(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
				max_age=(int)(round((numpy.percentile(possible_max_ages,75))))
				if max_age>numpy.percentile(lifespans,90):
					max_age=round(max_age=numpy.percentile(lifespans,90))		
			print(max_age)
		
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
			if len(miRNA)>1:
				miRNA=miRNA[0]+'_'+miRNA[1]
			else:
				miRNA=miRNA[0]
		keypoint_keys=list(worms[10].td._keys())		
		for key in keypoint_keys:
			if 'z' not in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
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
						filtered_worms=worms.filter(lambda worm: worm.lifespan > right_window)
						filtered_lifespans=filtered_worms.get_feature('lifespan')
						try:	
							uncontrolled_results=filtered_worms.regress(get_average_wrapper(left_window,right_window,key+'_z'),get_slope_wrapper(left_window,right_window,key+'_z'),target=target)
						except:
							continue
						correlations.append(uncontrolled_results.R2)
						windows.append([left_window,right_window])
				
				best_window=windows[correlations.index(max(correlations))]
				print(best_window)
	
				uncontrolled_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z'),get_slope_wrapper(best_window[0],best_window[1],key+'_z'),target=target)		
						
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_regression_on_slope_and_mean_sliding/'+miRNA+'_'+key+'.svg'

				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(uncontrolled_results.y_est+uncontrolled_results.resid,uncontrolled_results.y_est)
				ptext="p = " + (str)(round(p_value, 3))	
				color_vals = colorize.scale((uncontrolled_results.y_est+uncontrolled_results.resid)/24, output_max=1)
				colors = colorize.color_map(color_vals, uint8=False)
				plt.scatter((uncontrolled_results.y_est+uncontrolled_results.resid)/24,uncontrolled_results.y_est/24,c='indigo',alpha=.7)	
					
				(m,b) = numpy.polyfit((uncontrolled_results.y_est+uncontrolled_results.resid)/24, uncontrolled_results.y_est/24, 1)
				yp=numpy.polyval([m,b], (uncontrolled_results.y_est+uncontrolled_results.resid)/24)
				mew, mewmew=zip(*sorted(zip((uncontrolled_results.y_est+uncontrolled_results.resid)/24, yp)))
				mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
				slope_results=filtered_worms.regress(get_slope_wrapper(best_window[0],best_window[1],key+'_z'),target=target)
				average_results=filtered_worms.regress(get_average_wrapper(best_window[0],best_window[1],key+'_z'),target=target)
				correlations_to_write.append(slope_results.R2)
				correlations_to_write.append(average_results.R2)
				correlations_to_write.append(uncontrolled_results.R2)
				features_to_write.append(key+'_slope')
				features_to_write.append(key+'_mean')
				features_to_write.append(key+'_joint')
				features_to_write.append(key+'_joint_p_value')
				correlations_to_write.append((str)(round(p_value, 3)))
				features_to_write.append('best_window')
				correlations_to_write.append(best_window)
				features_to_write.append('n')
				correlations_to_write.append(len(uncontrolled_results.y_est))
				gtext='R2 = '+(str)(round(uncontrolled_results.R2,3))
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
		datafile.write_delimited('/Volumes/9karray/Kinser_Holly/all_keypoint_regression_on_slope_and_mean_sliding/'+miRNA+'_'+'_regression_correlations.tsv',[features_to_write]+[correlations_to_write])
	
def all_keypoint_correlation_plots_controlled(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>48]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age>numpy.percentile(lifespans,90):
				max_age=round(numpy.percentile(lifespans,90))		
			print(max_age)


		ages=[]
		
		for i in range(min_age,max_age+1):
			ages.append(i)	
		print(ages)
		keypoint_keys=list(worms[0].td._keys())
		for key in keypoint_keys:
			if 'z' in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				
				for i in range(0, len(keypoint_keys)):
					if 'z' in keypoint_keys[i] and 'age' not in keypoint_keys[i] and 'stage' not in keypoint_keys[i] and 'timepoint' not in keypoint_keys[i] and 'timestamp' not in keypoint_keys[i]:
						print(keypoint_keys[i])
						save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_correlation_plots_controlled/'+miRNA+'_'+key+'controlled_for_'+keypoint_keys[i]+'.svg'
						save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_correlation_plots_controlled/'+miRNA+'_'+key+'controlled_for_'+keypoint_keys[i]+'.svg'
			
						controlled_correlations=[]
						correlations=[]
						p_values=[]
						controlled_p_values=[]		
						for j in range(0,len(ages)-1):
							filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[j+1]*24)
							results=filtered_worms.regress(get_average_wrapper(ages[j]*24,ages[j+1]*24,key),target=target)
							controlled_results=filtered_worms.regress(get_average_wrapper(ages[j]*24,ages[j+1]*24,key),target=target,control_features=[get_average_wrapper(ages[j]*24,ages[j+1]*24,keypoint_keys[i])])
							correlations.append(results.R2)
							controlled_correlations.append(controlled_results.R2)
							slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X,results.y_est)
							slope, intercept,rvalue,controlled_p_value,stderror=scipy.stats.linregress(controlled_results.X,controlled_results.y_est)
							p_values.append(pvalue)
							controlled_p_values.append(controlled_p_value)


						p_x=[]
						p_y=[]
						for i in range(0,len(p_values)):
							if p_values[i]<.05 and correlations[i]>=.1:
								p_y.append(correlations[i]+.03)
								p_x.append(ages[1::][i])
						controlled_p_x=[]
						controlled_p_y=[]
						for i in range(0,len(controlled_p_values)):
							if controlled_p_values[i]<.05 and controlled_correlations[i]>=.1:
								controlled_p_y.append(controlled_correlations[i]+.03)
								controlled_p_x.append(ages[1::][i])						

						plt.scatter(p_x,p_y,marker=(6,2,0),c='navy',s=50)
						plt.scatter(controlled_p_x,controlled_p_y,marker=(6,2,0),c='gray',s=50)
						plt.scatter(ages[1::],correlations,c='navy',marker='o',s=50,edgecolor='navy')
						plt.scatter(ages[1::],controlled_correlations,c='gray',marker='o',s=50,edgecolor='gray')
						plt.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='navy')
						plt.plot(ages[1:ages.index(max_age)+1],controlled_correlations[0:ages.index(max_age)], c='gray')
						plt.title('P'+miRNA + 'GFP lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
						plt.xlabel('Age (day post-hatch)', fontdict=label_fontdict,fontsize=24)
						plt.ylabel('Coefficient of determination (R2)',fontdict=label_fontdict,fontsize=24)
						plt.ylim(0,.6)
						plt.margins(.15,.15)
						print('Saving '+save_name)	
						plt.savefig(save_name)
						plt.savefig(save_name_box)
						plt.gcf().clf
						plt.close()

def all_keypoint_scatter(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>2*24]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age>numpy.percentile(lifespans,20):
				max_age=numpy.percentile(lifespans,20)		
			print(max_age)
	

		ages=[]
		
		for i in range(min_age,max_age+1):
			ages.append(i)	
		print(ages)
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		for key in list(worms[0].td._keys()):
			if '_z' in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				

				for i in range(0,len(ages)-1):
					save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_scatter/'+miRNA+'_'+(str)(ages[i])+key+'.svg'
					save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_scatter/'+miRNA+'_'+(str)(ages[i])+key+'.svg'
					filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i+1]*24)
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key),target=target)
					lifespans=results.y_est+results.resid
					slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X, lifespans)
					color_vals = colorize.scale(results.X, output_max=1)
					colors = colorize.color_map(color_vals, uint8=False)	
					plt.scatter(results.X, lifespans,c=colors)
					(m,b) = numpy.polyfit(results.X, lifespans, 1)
					yp=numpy.polyval([m,b], lifespans)
					mew, mewmew=zip(*sorted(zip(lifespans, yp)))
					mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
					plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)
					ftext="$r^{2}$ = "+(str)(round(results.R2,3))
					more_text="n= " + (str)(len(lifespans))
					plt.title('Average ' +miRNA+'::GFP expression vs. lifespan',fontdict=title_fontdict)
					plt.xlabel('Actual lifespan (days)',fontdict=label_fontdict)
					plt.ylabel("Predicted lifespan (days)",fontdict=label_fontdict)
					plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
					plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
					plt.margins(.15,.15)
					print('Saving '+save_name)	
					plt.savefig(save_name)
					plt.savefig(save_name_box)
					plt.gcf().clf
					plt.close()				

					
def all_keypoint_correlation_plots(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			keys=list(worms[10].td._keys())			
			lifespans=worms.get_feature(target)
			possible_max_ages=[]
			for worm in worms:
				test_data=getattr(worm.td,keys[8])
				test_ages=worm.td.age
				bad_ages=test_ages[numpy.isnan(test_data)]
				max_age=bad_ages[bad_ages>2*24]
				if len(max_age>0):
					max_age=max_age[0]/24
					possible_max_ages.append(max_age)
				else:
					max_age=(int)(round(numpy.percentile(lifespans/24,10)))
					possible_max_ages.append(max_age)		
			max_age=(int)(round((numpy.percentile(possible_max_ages,50))))
			if max_age>numpy.percentile(lifespans,90):
				max_age=round(numpy.percentile(lifespans,90))		
			print(max_age)
	

		ages=[]
		
		for i in range(min_age,max_age+1):
			ages.append(i)	
		print(ages)
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		for key in list(worms[0].td._keys()):
			if '_z' in key and 'age' not in key and 'stage' not in key and 'timepoint' not in key and 'timestamp' not in key:
				print(key)
				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)
				if len(miRNA)>1:
					miRNA=miRNA[0]+'_'+miRNA[1]
				else:
					miRNA=miRNA[0]	
				save_name='/Volumes/9karray/Kinser_Holly/all_keypoint_correlation_plots/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_keypoint_correlation_plots/'+miRNA+'_'+key+'.svg'
			
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue

				correlations=[]
				p_values=[]	
				for i in range(0,len(ages)-1):
					filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i+1]*24)
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key),target=target)
					correlations.append(results.R2)
					lifespans=results.y_est+results.resid
					slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(lifespans,results.y_est)
					p_values.append(pvalue)
	

				correlations_towrite.append(correlations)
				features_towrite.append(key)

				p_x=[]
				p_y=[]
				for i in range(0,len(p_values)):
					if p_values[i]<.05 and correlations[i]>=.1:
						p_y.append(correlations[i]+.03)
						p_x.append(ages[1::][i])				
				plt.scatter(p_x,p_y,marker=(6,2,0),c='navy',s=50)
				plt.scatter(ages[1::],correlations,c='navy',marker='o',s=50,edgecolor='navy')
				plt.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='navy')
				plt.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='navy',linestyle='--')	
				plt.title('P'+miRNA + '::GFP lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
				plt.xlabel('Age (day post-hatch)', fontdict=label_fontdict, fontsize=24)
				plt.ylabel('Coefficient of determination (R2)',fontdict=label_fontdict, fontsize=24)
				plt.ylim(0,.6)
				plt.margins(.15,.15)
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
		datafile.write_delimited(subdirectory+'/correlations.tsv',[features_towrite]+rows)			
def all_wild_type_correlation_plots(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			try:			
				lifespans=worms.get_feature(target)
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
			try:

				miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
				save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots/'+miRNA+'_'+key+'.svg'
				save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots/'+miRNA+'_'+key+'.svg'
			
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue
				correlations=[]
				p_values=[]	
				for i in range(0,len(ages)-1):
					filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i]*24)
					results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z'),target=target)
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
				plt.ylabel('Coefficient of determination (r2)',fontdict=label_fontdict)
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
		datafile.write_delimited(subdirectory+'/correlations.tsv',[features_towrite]+rows)
def all_wild_type_F_stat(root_folder,overwrite=True,min_age=3,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			try:			
				lifespans=worms.get_feature(target)
				max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				extended_max_age=(int)(round(numpy.percentile(lifespans/24,50)))
			except:
				continue	
		ages=[]
		
			

	
		for key,value in gfp_measures.items():
			filtered_worms=worms.filter(lambda worm: worm.lifespan > max_age)
	
			data=filtered_worms.get_time_range(key+'_z',min_age,max_age*24)
				

			total_subjects=0
			grand_mean=0
			for i in range(0, len(data)):
				for j in range(0, len(data[i])):
					total_subjects=total_subjects+1
					grand_mean=grand_mean+data[i][j][1]


			try:
				grand_mean=grand_mean/total_subjects
			except:
				continue		

			between_groups_df=len(filtered_worms)-1
			within_groups_df=total_subjects-between_groups_df
			between_groups_squares=[]
			within_groups_squares=[]
			sample_means=[]
			group_n=[]
			for i in range(0, len(data)):
				sample_means.append(numpy.mean(data[i][:,1]))
				group_n.append(len(data[i]))
			for i in range(0, len(data)):
				sum=0
				for j in range(0, len(data[i])):
					sum=(data[i][j][1]-grand_mean)**2+sum 
					within_groups_squares.append(sum*(group_n[i]-1))

			for i in range(0, len(sample_means)):
				between_groups_squares.append(group_n[i]*(((sample_means[i]-grand_mean)**2)))

			between_group_sum_of_squares=numpy.sum(between_groups_squares)/between_groups_df
			within_group_sum_of_squares=numpy.sum(within_groups_squares)/within_groups_df
			f_stat=between_group_sum_of_squares/within_group_sum_of_squares
			print(subdirectory)
			print(key)
			print(between_group_sum_of_squares)
			print(within_group_sum_of_squares)
			print(f_stat)			

def all_wild_type_heat_maps(root_folder,overwrite=True,min_age=3,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			try:			
				lifespans=worms.get_feature(target)
				max_age=(int)(round(numpy.percentile(lifespans/24,10)))
				extended_max_age=(int)(round(numpy.percentile(lifespans/24,50)))
			except:
				continue	
		ages=[]
		
		for i in range(min_age,extended_max_age+1):
			ages.append(i)	

	
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_heat_maps/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_heat_maps/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			worms.sort('lifespan', reverse=True)
			trimmed_data=[[]for worm in worms]
			df_ages=[[]for worm in worms]
			try:
				data=worms.get_time_range(key+'_z',ages[0]*24,ages[-1]*24)
			except:
				continue	
				
			for i in range(0, len(data)):
				for j in range(0, len(data[i])):
					trimmed_data[i].append(data[i][j][1])
					df_ages[i].append(round(data[i][j][0]/24,1))
			max_ages=[]		
			for i in range(0, len(df_ages)):
				max_ages.append(len(df_ages[i]))		
			cols=df_ages[max_ages.index(max(max_ages))]
			index=worms
			df=pd.DataFrame(trimmed_data,index=index,columns=cols)
		
			seaborn.heatmap(df,vmin=-3,vmax=3,linecolor='None',yticklabels=False, cmap=seaborn.diverging_palette(240, 10, n=9),cbar_kws={'label': 'Expression Z score'})			
			print('Saving '+save_name)	
			plt.savefig(save_name)
			plt.savefig(save_name_box)
			plt.gcf().clf
			plt.close()
		
	
def all_wild_type_scatter_plots(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			try:			
				lifespans=worms.get_feature("lifespan")
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
			
				if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
					continue
				filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i]*24)
				lifespans=filtered_worms.get_feature('lifespan')
			
				results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z'),target=target)
					
				slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X,results.y_est)
				color_vals = colorize.scale(lifespans, output_max=1)
				colors = colorize.color_map(color_vals, uint8=False)
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
#generate correlation plots for reporters that correlate with lifespan. window is established based on 10th percentile of lifespan. extended window based on 90th percentile.
def all_wild_type_correlation_plots_with_survival_curve(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			try:			
				lifespans=worms.get_feature('lifespan')
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_survival_curve/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_survival_curve/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			correlations=[]
			p_values=[]		
			for i in range(0,len(ages)-1):
				filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i]*24)
				results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z'),target=target)
				correlations.append(results.R2)
				slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X,results.y_est)
				p_values.append(pvalue)

			correlations_towrite.append(correlations)
			features_towrite.append(key)
			p_x=[]
			p_y=[]
			for i in range(0,len(p_values)):
				if p_values[i]<.05 and correlations[i]>=.1:
					p_y.append(correlations[i]+.03)
					p_x.append(ages[1::][i])
			

			lifespans=worms.get_feature('lifespan')/24
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
			ax1.set_xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict)
			ax1.set_ylabel('Fraction alive (%)', fontdict=label_fontdict, color=color)
			ax1.tick_params(axis='y', labelcolor=color)	
			ax1.plot(days,percent_alive,color=color,alpha=.5)
			ax2 = ax1.twinx()
			color = 'green'
			ax2.set_ylabel('Coefficient of determination (r2)',fontdict=label_fontdict, color=color)
			ax2.scatter(p_x,p_y,marker=(6,2,0),c='green',s=50)
			ax2.scatter(ages[1::],correlations,c='green',marker='o',s=25,edgecolor='green')
			ax2.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='green')
			ax2.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='green',linestyle='--')
			ax2.tick_params(axis='y', labelcolor=color)	
			ax2.set_ylim(0,.4)		
		
			plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+target, y=1.05, fontdict=title_fontdict)
		
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
		datafile.write_delimited(subdirectory+'/correlations.tsv',[features_towrite]+rows)
def all_wild_type_correlation_plots_with_auto_with_survival_curve(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]
			try:			
				lifespans=worms.get_feature(target)
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
			save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlation_plots_with_auto_with_survival_curve/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlation_plots_with_auto_with_survival_curve/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			correlations=[]
			auto_correlations=[]
			length_correlations=[]
			p_values=[]
			auto_p_values=[]
			length_p_values=[]		
			for i in range(0,len(ages)-1):
				filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i]*24)
				results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z'),target=target)
				results_auto=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,'green_yellow_excitation_autofluorescence_percentile_95_z'),target=target)
				results_length=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,'length'),target=target)
				correlations.append(results.R2)
				auto_correlations.append(results_auto.R2)
				length_correlations.append(results_length.R2)
				slope, intercept,rvalue,pvalue,stderror=scipy.stats.linregress(results.X,results.y_est)
				p_values.append(pvalue)
				slope, intercept,rvalue,pvalue_auto,stderror=scipy.stats.linregress(results_auto.X,results_auto.y_est)
				auto_p_values.append(pvalue_auto)
				slope, intercept,rvalue,pvalue_length,stderror=scipy.stats.linregress(results_length.X,results_length.y_est)
				length_p_values.append(pvalue_length)

			correlations_towrite.append(correlations)
			features_towrite.append(key)
			p_x=[]
			p_y=[]
			p_x_auto=[]
			p_y_auto=[]
			p_x_length=[]
			p_y_length=[]
			for i in range(0,len(p_values)):
				if p_values[i]<.05 and correlations[i]>=.1:
					p_y.append(correlations[i]+.02)
					p_x.append(ages[1::][i])
				if auto_p_values[i]<.05 and auto_correlations[i]>=.1:
					p_x_auto.append(ages[1::][i])
					p_y_auto.append(auto_correlations[i]+.02)
				if length_p_values[i]<.05 and length_correlations[i]>=.1:
					p_x_length.append(ages[1::][i])
					p_y_length.append(length_correlations[i]+.02)		
			

			lifespans=worms.get_feature('lifespan')/24
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
			ax1.set_xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict)
			ax1.set_ylabel('Fraction alive (%)', fontdict=label_fontdict, color=color)
			ax1.tick_params(axis='y', labelcolor=color)	
			ax1.plot(days,percent_alive,color=color,alpha=.5)
			ax2 = ax1.twinx()
			color = 'green'
			ax2.set_ylabel('Coefficient of determination ('+"$r^{2}$"+')',fontdict=label_fontdict, color=color)
			ax2.scatter(p_x,p_y,marker=(6,2,0),c='green',s=25)
			ax2.scatter(ages[1::],correlations,c='green',marker='o',s=50,edgecolor='green')
			ax2.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='green')
			ax2.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='green',linestyle='--')
			ax2.tick_params(axis='y', labelcolor=color)	
			ax2.set_ylim(0,.4)
			ax2.scatter(p_x_auto,p_y_auto,marker=(6,2,0),c='red',s=25)
			ax2.scatter(ages[1::],auto_correlations,c='red',marker='o',s=50,edgecolor='red')
			ax2.plot(ages[1:ages.index(max_age)+1],auto_correlations[0:ages.index(max_age)], c='red')
			ax2.plot(ages[ages.index(max_age)::],auto_correlations[ages.index(max_age)-1::],c='red',linestyle='--')
			ax2.scatter(p_x_length,p_y_length,marker=(6,2,0),c='blue',s=25)
			ax2.scatter(ages[1::],length_correlations,c='blue',marker='o',s=50,edgecolor='blue')
			ax2.plot(ages[1:ages.index(max_age)+1],length_correlations[0:ages.index(max_age)], c='blue')
			ax2.plot(ages[ages.index(max_age)::],length_correlations[ages.index(max_age)-1::],c='blue',linestyle='--')

			red_patch = mpatches.Patch(color='red', label='autofluorescence')
			green_patch = mpatches.Patch(color='green',label='GFP')
			blue_patch=mpatches.Patch(color='blue',label='length')
			plt.legend(handles=[red_patch,green_patch,blue_patch])		
		
			plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+target, y=1.05, fontdict=title_fontdict)
		
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
		datafile.write_delimited(subdirectory+'/correlations.tsv',[features_towrite]+rows)		
def all_mutant_regression_on_slope_and_mean(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
				worm.lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]	

		mut_lifespans=mut_worms.get_feature(target)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans,10)))
		control_lifespans=control_worms.get_feature(target)
		control_max_age=(int)(round(numpy.percentile(control_lifespans,10)))
		
		
		
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_mutant_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_mutant_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
		
			filtered_mut_worms=mut_worms.filter(lambda worm: worm.lifespan > mut_max_age)
			filtered_control_worms=control_worms.filter(lambda worm: worm.lifespan > control_max_age)
			mut_results=filtered_mut_worms.regress(get_average_wrapper(min_age,mut_max_age,key+'_z'),get_slope_wrapper(min_age,mut_max_age,key+'_z'),target=target)
			control_results=filtered_control_worms.regress(get_average_wrapper(min_age,control_max_age,key+'_z'),get_slope_wrapper(min_age,control_max_age,key+'_z'),target=target)
			mut_lifespans=filtered_mut_worms.get_feature(target)
			control_lifespans=filtered_control_worms.get_feature(target)
		
			
			orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
			plt.legend(handles=[orangered_patch,indigo_patch])
			slope, intercept, r_value, mut_p_value, std_err=scipy.stats.linregress(mut_lifespans,mut_results.y_est)
			mut_ptext="p = " + (str)(round(mut_p_value, 3))
			slope, intercept, r_value, control_p_value, std_err=scipy.stats.linregress(control_lifespans,control_results.y_est)
			control_ptext="p = " + (str)(round(control_p_value, 3))
			plt.figtext(.15,.75,mut_ptext, fontsize=20,ha='left',color='orangered')
			plt.figtext(.15,.65,control_ptext, fontsize=20,ha='left', color='indigo')	
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
			mut_ftext="$r^{2}$ = "+(str)(round(mut_results.R2,3))
			control_ftext="$r^{2}$ = "+(str)(round(control_results.R2,3))
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
def all_daf16_regression_on_slope_and_mean(root_folder,overwrite=True,min_age=3*24,target='lifespan'):
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
				worm.lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]	

		mut_lifespans=mut_worms.get_feature(target)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans,10)))
		control_lifespans=control_worms.get_feature(target)
		control_max_age=(int)(round(numpy.percentile(control_lifespans,10)))
		print(mut_max_age)
		
		for key,value in gfp_measures.items():
			try:
				mut_worms.get_time_range(key,24,numpy.inf)
			except:
				continue
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[1]
			save_name='/Volumes/9karray/Kinser_Holly/all_daf16_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_daf16_regression_on_slope_and_mean/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
		
			filtered_mut_worms=mut_worms.filter(lambda worm: worm.lifespan > mut_max_age)
			filtered_control_worms=control_worms.filter(lambda worm: worm.lifespan > control_max_age)
			mut_results=filtered_mut_worms.regress(get_average_wrapper(min_age,mut_max_age,key+'_z'),get_slope_wrapper(min_age,mut_max_age,key+'_z'),target=target)
			control_results=filtered_control_worms.regress(get_average_wrapper(min_age,control_max_age,key+'_z'),get_slope_wrapper(min_age,control_max_age,key+'_z'),target=target)
			mut_lifespans=filtered_mut_worms.get_feature(target)
			control_lifespans=filtered_control_worms.get_feature(target)
			
			orangered_patch = mpatches.Patch(color='orangered', label='daf-16')
			indigo_patch = mpatches.Patch(color='indigo',label='wild-type')
			plt.legend(handles=[orangered_patch,indigo_patch])
			slope, intercept, r_value, mut_p_value, std_err=scipy.stats.linregress(mut_lifespans,mut_results.y_est)
			slope, intercept, r_value, control_p_value, std_err=scipy.stats.linregress(control_lifespans,control_results.y_est)	
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
			plt.xlabel('Lifespan (hours)',fontdict=label_fontdict)
			plt.ylabel("Expression-predicted lifespan (Days)",fontdict=label_fontdict,fontsize=24)
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


def all_mutant_correlation_plots(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
				worm.lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]	

		mut_lifespans=mut_worms.get_feature(target)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans/24,10)))
		mut_extended_max_age=(int)(round(numpy.percentile(mut_lifespans/24,75)))
		control_lifespans=control_worms.get_feature(target)
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
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			mut_correlations=[]
			control_correlations=[]
			mut_p_values=[]
			control_p_values=[]	
			for i in range(0,len(mut_ages)-1):
				filtered_mut_worms=mut_worms.filter(lambda worm: worm.lifespan > mut_ages[i]*24)
				mut_results=filtered_mut_worms.regress(get_average_wrapper(mut_ages[i]*24,mut_ages[i+1]*24,key+'_z'),target=target)
				mut_correlations.append(mut_results.R2)
				slope, intercept,rvalue,mut_pvalue,stderror=scipy.stats.linregress(mut_results.X,mut_results.y_est)
				mut_p_values.append(mut_pvalue)
			for i in range(0,len(control_ages)-1):
				filtered_control_worms=control_worms.filter(lambda worm: worm.lifespan > control_ages[i]*24)
				control_results=filtered_control_worms.regress(get_average_wrapper(control_ages[i]*24,control_ages[i+1]*24,key+'_z'),target=target)
				control_correlations.append(control_results.R2)
				slope, intercept,rvalue,control_pvalue,stderror=scipy.stats.linregress(control_results.X,control_results.y_est)
				control_p_values.append(control_pvalue)	

			mut_correlations_towrite.append(mut_correlations)
			control_correlations_towrite.append(control_correlations)
			features_towrite.append(key)

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
			plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+target, y=1.05, fontdict=title_fontdict)
			plt.xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict)
			plt.ylabel('Coefficient of determination ('+"$r^{2}$"+')',fontdict=label_fontdict)
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
		datafile.write_delimited(subdirectory+'/mutant_correlations.tsv',[features_towrite]+rows)
		n=len(control_ages)
		rows=[[] for _ in range(n)]
		for col in control_correlations_towrite:
			for row, colval in zip(rows,col):
				row.append(colval)
		datafile.write_delimited(subdirectory+'/control_correlations.tsv',[features_towrite]+rows)
def all_daf16_correlation_plots(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
				worm.lifespan = worm.lifespan/ mut_median_lifespans[exp] * mut_median_lifespans[mut_exp_dirs[0]]
				worm.td.age=worm.td.age/mut_median_lifespans[exp]*mut_median_lifespans[mut_exp_dirs[0]]
			for worm in control_worms:
				exp = worm.name.split()[0]
				worm.lifespan = worm.lifespan/ control_median_lifespans[exp] * control_median_lifespans[control_exp_dirs[0]]
				worm.td.age=worm.td.age/control_median_lifespans[exp]*control_median_lifespans[control_exp_dirs[0]]	

		mut_lifespans=mut_worms.get_feature(target)
		mut_max_age=(int)(round(numpy.percentile(mut_lifespans/24,10)))
		mut_extended_max_age=(int)(round(numpy.percentile(mut_lifespans/24,75)))
		control_lifespans=control_worms.get_feature(target)
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
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			mut_correlations=[]
			control_correlations=[]
			mut_p_values=[]
			control_p_values=[]
			try:	
				for i in range(0,len(mut_ages)-1):
					filtered_mut_worms=mut_worms.filter(lambda worm: worm.lifespan > mut_ages[i]*24)
					mut_results=filtered_mut_worms.regress(get_average_wrapper(mut_ages[i]*24,mut_ages[i+1]*24,key+'_z'),target=target)
					mut_correlations.append(mut_results.R2)
					slope, intercept,rvalue,mut_pvalue,stderror=scipy.stats.linregress(mut_results.X,mut_results.y_est)
					mut_p_values.append(mut_pvalue)
				for i in range(0,len(control_ages)-1):
					filtered_control_worms=control_worms.filter(lambda worm: worm.lifespan > control_ages[i]*24)
					control_results=filtered_control_worms.regress(get_average_wrapper(control_ages[i]*24,control_ages[i+1]*24,key+'_z'),target=target)
					control_correlations.append(control_results.R2)
					slope, intercept,rvalue,control_pvalue,stderror=scipy.stats.linregress(control_results.X,control_results.y_est)
					control_p_values.append(control_pvalue)

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

				mut_correlations_towrite.append(mut_correlations)
				control_correlations_towrite.append(control_correlations)
				features_towrite.append(key)				
				orangered_patch = mpatches.Patch(color='orangered', label='daf-16')
				indigo_patch = mpatches.Patch(color='indigo',label='wild-type')
				plt.legend(handles=[orangered_patch,indigo_patch])
				plt.scatter(mut_p_x,mut_p_y,marker=(6,2,0),c='orangered',s=50)
				plt.scatter(control_p_x,control_p_y,marker=(6,2,0),c='indigo',s=50)
				plt.scatter(mut_ages[1::],mut_correlations,c='orangered',marker='o',s=50,edgecolor='orangered')
				plt.scatter(control_ages[1::],control_correlations,c='indigo',marker='o',s=50,edgecolor='indigo')
				plt.plot(mut_ages[1:mut_ages.index(mut_max_age)+1],mut_correlations[0:mut_ages.index(mut_max_age)], c='orangered',linestyle='--')
				plt.plot(control_ages[1:control_ages.index(control_max_age)+1],control_correlations[0:control_ages.index(control_max_age)], c='indigo')
				plt.plot(mut_ages[mut_ages.index(mut_max_age)::],mut_correlations[mut_ages.index(mut_max_age)-1::],c='orangered',alpha=.5,linestyle='--')
				plt.plot(control_ages[control_ages.index(control_max_age)::],control_correlations[control_ages.index(control_max_age)-1::],c='indigo',alpha=.5)	
				plt.title('P'+miRNA + '::GFP lifespan-predictive value over time', y=1.05, fontdict=title_fontdict)
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
			except:
				continue	
			for col in mut_correlations_towrite:
				for row, colval in zip(rows,col):
					row.append(colval)
			datafile.write_delimited(subdirectory+'/mutant_correlations.tsv',[features_towrite]+rows)
			n=len(control_ages)
			rows=[[] for _ in range(n)]
			for col in control_correlations_towrite:
				for row, colval in zip(rows,col):
					row.append(colval)
				
		datafile.write_delimited(subdirectory+'/control_correlations.tsv',[features_towrite]+rows)				
def all_nonmir_correlation_plots(root_folder,overwrite=True,min_age=2,target='lifespan'):
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
		else:
			if len(exp_dirs)>1:
				median_lifespans={}
				groups=worms.group_by([w.name.split()[0] for w in worms])	
	
				for exp, wormies in groups.items():
					median_lifespans[exp] = numpy.median(wormies.get_feature('lifespan'))
				for worm in worms:
						exp = worm.name.split()[0]
						worm.lifespan = worm.lifespan/ median_lifespans[exp] * median_lifespans[exp_dirs[0]]
						worm.td.age=worm.td.age/median_lifespans[exp]*median_lifespans[exp_dirs[0]]

			lifespans=worms.get_feature(target)
			max_age=(int)(round(numpy.percentile(lifespans/24,10)))
			extended_max_age=(int)(round(numpy.percentile(lifespans/24,75)))

		ages=[]
		
		for i in range(min_age,extended_max_age+1):
			ages.append(i)	
		print(ages)
		features_towrite=['age']
		correlations_towrite=[ages[1::]]
		for key,value in gfp_measures.items():
			miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
			save_name='/Volumes/9karray/Kinser_Holly/all_nonmir_correlation_plots/'+miRNA+'_'+key+'.svg'
			save_name_box='/Users/pincuslab/Box/miRNA Data/all_nonmir_correlation_plots/'+miRNA+'_'+key+'.svg'
			
			if (pathlib.Path(save_name).exists()==True) and (overwrite==False):
				continue
			correlations=[]	
			for i in range(0,len(ages)-1):
				filtered_worms=worms.filter(lambda worm: worm.lifespan > ages[i]*24)
				results=filtered_worms.regress(get_average_wrapper(ages[i]*24,ages[i+1]*24,key+'_z'),target=target)
				correlations.append(results.R2)

			correlations_towrite.append(correlations)
			features_towrite.append(key)				
			
			plt.scatter(ages[1::],correlations,c='navy',marker='o',s=50,edgecolor='navy')
			plt.plot(ages[1:ages.index(max_age)+1],correlations[0:ages.index(max_age)], c='navy')
			plt.plot(ages[ages.index(max_age)::],correlations[ages.index(max_age)-1::],c='navy',linestyle='--')
			plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+target, y=1.05, fontdict=title_fontdict)
			plt.xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict)
			plt.ylabel('Coefficient of determination (R2)',fontdict=label_fontdict,fontsize=24)
			plt.ylim(0,.6)
			plt.margins(.15,.15)
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
		datafile.write_delimited(subdirectory+'/correlations.tsv',[features_towrite]+rows)		
#plot survival curves for reporters that correlate with lifespan, with correlating window (r^2>.1) shaded in

def all_daf16_survival_curves (root_folder):
	subdirectories=glob(root_folder+'/*')
	for subdirectory in subdirectories:
		mut_exp_dirs=sorted(glob(subdirectory+'/mutant/*'))
		control_exp_dirs=sorted(glob(subdirectory+'/control/*'))
		mut_worms=process_worms(mut_exp_dirs, prefixes=[dir+' ' for dir in mut_exp_dirs])
		control_worms=process_worms(control_exp_dirs, prefixes=[dir+' ' for dir in control_exp_dirs])
		miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[1]

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

		plt.plot(mut_days,mut_percent_alive,color='orangered',linestyle='--')
		plt.plot(control_days,control_percent_alive,color='indigo')	
		orangered_patch = mpatches.Patch(color='orangered', label='mutant', alpha=.7)
		indigo_patch = mpatches.Patch(color='indigo',label='wild-type',alpha=.7)
		plt.legend(handles=[orangered_patch,indigo_patch])
		plt.xlabel("Age (days)",fontdict=label_fontdict)
		plt.ylabel("Survival (%)",fontdict=label_fontdict)
		more_text="median mutant lifespan = " + (str)((round(numpy.median(mut_lifespans),1))) + " days"
		more_text_ctr="median control lifespan = " + (str)((round(numpy.median(control_lifespans),1))) + " days"
		even_more_text="n = "+(str)(len(mut_lifespans))
		even_more_text_ctr="n = "+(str)(len(control_lifespans))
		plt.figtext(.15, .25, more_text, fontsize=15, ha='left',color='orangered')
		plt.figtext(.15, .22, more_text_ctr, fontsize=15, ha='left',color='indigo')
		plt.figtext(.15, .19, even_more_text, fontsize=15, ha='left',color='orangered')
		plt.figtext(.15, .16, even_more_text_ctr, fontsize=15, ha='left',color='indigo')
		save_name='/Volumes/9karray/Kinser_Holly/all_daf16_survival_curves/'+miRNA+'_survival_curve.svg'
		save_name_box='/Users/pincuslab/Box/miRNA Data/all_daf16_survival_curves/'+miRNA+'_survival_curve.svg'
		plt.savefig(save_name)
		plt.savefig(save_name_box)
		plt.gcf().clf
		plt.close()	
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
		plt.ylabel("Fraction Alive (%)",fontdict=label_fontdict)
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



def all_wild_type_correlated_survival_curves(root_folder):
	subdirectories=os.listdir(root_folder)
	subdirectories=[root_folder+'/'+subdirectory for subdirectory in subdirectories if pathlib.Path(root_folder+'/'+subdirectory).is_dir()]
	for subdirectory in subdirectories:

		exp_dirs=os.listdir(subdirectory)
		exp_dirs=[subdirectory+'/'+exp_dir for exp_dir in exp_dirs if pathlib.Path(subdirectory+'/'+exp_dir).is_dir()]
		worms=process_worms(exp_dirs, prefixes=[dir+' ' for dir in exp_dirs])
		miRNA=re.findall(r"\w\w\w-\d+",subdirectory)[0]
		if pathlib.Path(subdirectory+'/'+'correlations.tsv').exists():
			header,parameters=datafile.read_delimited(subdirectory+'/'+'correlations.tsv')
			parameters=list(parameters)
			summed_correlations=dict.fromkeys(header[3:-2])
		for key, entry in summed_correlations.items():
			index=header.index(key)
			summed_correlations[key]=numpy.sum(parameters[i][index] for i in range(0,len(parameters)))
		print(miRNA)	
		print(summed_correlations)
	
		max_corr=max(list(summed_correlations.values()))
		best_parameter=max(summed_correlations,key=summed_correlations.get)
		correlating_ages=[]
		for i in range(0, len(parameters)):
			correlation=parameters[i][header.index(best_parameter)]
			if correlation>.1:
				correlating_ages.append(parameters[i][0])
		print(best_parameter)
		print(correlating_ages)			
		lifespans=worms.get_feature('lifespan')/24
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
		color_vals = colorize.scale(days, output_max=1)
		colors = colorize.color_map(color_vals, uint8=False)
		plt.plot(days,percent_alive,color='gray',alpha=.3)	
		plt.scatter(days,percent_alive,color=colors)
		if len(correlating_ages)>0:
			x=numpy.linspace(correlating_ages[0],correlating_ages[-1],len(percent_alive))
			y1=numpy.zeros(len(percent_alive))+100
			plt.fill_between(x,y1, color='gray', alpha=.5)
		plt.xlabel("Age (days)",fontdict=label_fontdict)
		plt.ylabel("Fraction Alive (%)",fontdict=label_fontdict)
		more_text="median lifespan = " + (str)((round(numpy.median(lifespans),1))) + " days"
		even_more_text="n = "+(str)(len(lifespans))
		plt.figtext(.15, .2, more_text, fontsize=20, ha='left', weight='bold')
		plt.figtext(.15,.25,best_parameter,fontsize=20,ha='left',weight='bold')
		plt.figtext(.15, .15, even_more_text, fontsize=20, ha='left', weight='bold')
		save_name='/Volumes/9karray/Kinser_Holly/all_wild_type_correlated_survival_curves/'+miRNA+'_survival_curve.svg'
		save_name_box='/Users/pincuslab/Box/miRNA Data/all_wild_type_correlated_survival_curves/'+miRNA+'_survival_curve.svg'
		plt.savefig(save_name)
		plt.savefig(save_name_box)
		plt.gcf().clf
		plt.close()			

	




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

def z_scoring_keypoint(root_folder, age_feature='age',overwrite=True):
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
			for key in list(worms[0].td._keys()):
				if 'age' not in key and 'timepoint' not in key and 'timestamp' not in key:
	
					try:
						worms.z_transform(key,age_feature=age_feature)
					except:
						continue	
			print('done transforming '+exp_dir)
			worms.write_timecourse_data(save_dir)			

#z transform all files in a directory and save out file with the same name (will keep old measurements)
def z_scoring(root_folder, age_feature='age',gfp_measures=gfp_measures,overwrite=True):
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
			worms.z_transform('length',age_feature=age_feature)			
			print('done transforming '+exp_dir)
			worms.write_timecourse_data(save_dir)			

			
def process_worms(paths,prefixes=['']):
	wormies=[]
	for path, prefix in zip(paths,prefixes):
		worms=worm_data.read_worms(path+'/*.tsv',name_prefix=prefix)
		wormies=wormies+worms
	
	return wormies