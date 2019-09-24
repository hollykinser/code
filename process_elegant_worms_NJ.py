from elegant import worm_data
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from zplib.image import colorize
import numpy
import scipy.stats
import time
import statsmodels.api as sm
from pandas import DataFrame
from sklearn import linear_model


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

plt.rcParams['axes.labelpad']=15.0
plt.rcParams['savefig.transparent']=True
plt.rcParams['legend.labelspacing']=.5
plt.rcParams['lines.linewidth']=3
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['xtick.major.pad'] = 9
plt.rcParams['ytick.major.pad'] = 9
plt.rcParams['lines.markersize']= 10

TIME_UNITS = dict(days=24, hours=1, minutes=1/60, seconds=1/3600)

label_fontdict={'size':20,'family':'arial'}
title_fontdict={'size':22, 'weight':'bold','family':'arial'}
text_fontdict={'size':20,'weight':'bold','family':'arial'}

gfp_measures={
'gfp_expression_fraction':'expression area fraction', 
'gfp_expression_mean':'expression area mean',
'gfp_expression_median':'expression area median',
'gfp_expression_sum':'summed expression area',
'gfp_high_expression_fraction':'high expression area fraction',
'gfp_high_expression_mean':'high expression area mean',
'gfp_high_expression_median':'high expression area median',
'gfp_high_expression_sum':'summed high expression area',
'gfp_mean':'mean intensity',
'gfp_median':'median intensity',
'gfp_maximum':'maximum intensity',
'gfp_over_99_mean':'mean of pixels over 99th percentile intensity',
'gfp_over_99_median':'median of pixels over 99th percentile intensity',
'gfp_over_99_sum':'sum of pixels over 99th percentile intensity',
'gfp_percentile_95':'95th percentile intensity',
'gfp_percentile_99':'99th percentile intensity',
'gfp_sum':'summed intensity',
'green_yellow_excitation_autofluorescence_percentile_95': '95th percentile intensity of autofluorescence'

}

numpy_functions={'Mean':numpy.mean,'Median':numpy.median,'Maximum':numpy.max,'Minimum':numpy.min}
plt.style.use('seaborn-white')

#top level functions: plot_all_scatters will make a scatter plot for each day timepoint with individual expression against lifespan
#plot_all_correlations will plot correlations from the scatter plots over time


#e.g. age_feature='age'. Default for other functions (plot_all_scatters and plot_all_correlations) is to use z_scored data, so perform this first.
def z_scoring(worms, age_feature='age',gfp_measures=gfp_measures):
	groups=worms.group_by([w.name.split()[0] for w in worms])
	for exp, wormies in groups.items():
		for feature in gfp_measures:
			wormies.z_transform(feature,age_feature=age_feature)

#paths is list containing paths to folder containing tsvs
def process_worms(paths,prefixes=''):
	wormies=[]
	for path, prefix in zip(paths,prefixes):
		worms=worm_data.read_worms(path+'/*.tsv',name_prefix=prefix)
		wormies=wormies+worms
	
	return wormies

#min_age is in hours. I usually do 48 or 72 hours. max_age will be calculated to be timepoint when 90% of the animals are alived unless otherwise specified 
#save_dir is path to directory where you want figures to save to
def plot_all_scatters(worms,min_age,save_dir,age_feature='lifespan',function='Mean',gfp_measures=gfp_measures,max_age=0,miRNA='',control_for=None):
	
	ages=[]
	lifespans=worms.get_feature('lifespan')

	if max_age==0:
		max_age=(int)(numpy.round(numpy.percentile(lifespans,10)/24))
	for i in range(min_age,max_age*24+1,24):
		ages.append(i)
		print(ages)

	for key, value in gfp_measures.items():
		for j in range(0, len(ages)-1):

			plot_scatter(worms,ages[j],ages[j+1],key,save_dir,age_feature,function,miRNA=miRNA,gfp_measures=gfp_measures,control_for=control_for)

def plot_scatter(worms, min_age, max_age,feature,save_dir,age_feature,function,miRNA='',gfp_measures=gfp_measures,control_for=None):
	if control_for is not None:
		data = worms.get_time_range(feature+'_z', min_age, max_age)
		control_data=worms.get_time_range(control_for+'_z', min_age, max_age)
		lifespans=worms.get_feature(age_feature)/24
		spans=worms.get_feature(age_feature)/24
		save_name=save_dir+'/'+feature + ' '+function+' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24)) +'_'+age_feature+'_'+control_for+'.svg'
		n_function=numpy_functions[function]
		averages=[]
		new_lifespans=[]
		control_averages=[]
		for i in range(0, len(data)):
			if lifespans[i]>=max_age/24 and len(data[i])>1:
				average=n_function(data[i][:,1])
				control_average=n_function(control_data[i][:,1])
				if -3<average<3:
					averages.append(average)
					new_lifespans.append(spans[i])
					control_averages.append(control_average)

		remaining_lifespans=[i-max_age/24 for i in new_lifespans]

		averages=numpy.asarray(averages)
		control_averages=numpy.asarray(control_averages)
		slope_x, intercept_x,rvalue_x,pvalue_x,stderror_x=scipy.stats.linregress(x=control_averages,y=averages)
		slope_y,intercept_y,rvalue_y,pvalue_y,stderror_y=scipy.stats.linregress(x=control_averages,y=remaining_lifespans)
		obsvalues_x=averages
		predvalues_x=control_averages*slope_x+intercept_x
		residuals_x=predvalues_x-obsvalues_x
		obsvalues_y=remaining_lifespans
		predvalues_y=control_averages*slope_y+intercept_y
		residuals_y=predvalues_y-obsvalues_y
		pearson=scipy.stats.pearsonr(residuals_x,residuals_y)
		spearman=scipy.stats.spearmanr(residuals_x,residuals_y)
	
		color_vals = colorize.scale(remaining_lifespans, output_max=1)
		colors = colorize.color_map(color_vals, uint8=False)
		#(m,b) = polyfit(residuals_x, residuals_y, 1)
		#yp=polyval([m,b], residuals_x)
		plt.scatter(residuals_x, residuals_y, c='gray')
		#plot(residuals_x,yp, c='gray',alpha=.7)

		plt.xlabel(function+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
		plt.ylabel("Days of life remaining",fontdict=label_fontdict)
		plt.title(function+' '+ 'P'+miRNA +'::GFP expression at '+(str)((int)(max_age/24))+' dph ' +age_feature,fontdict=title_fontdict)
		ymin, ymax=plt.ylim()
		plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


		(m,b) = numpy.polyfit(residuals_x, residuals_y, 1)
		yp=numpy.polyval([m,b], residuals_x)
		mew, mewmew=zip(*sorted(zip(residuals_x, yp)))
		mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
		plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)

	
		if pearson[1]<.00001:
			p="p<.00001"
		else:
			p="p=" + ''+ (str)(round(pearson[1],3))
		if spearman[1]<.00001:
			spearman_p="p<.00001"
		else:
			spearman_p="p=" + '' + (str)(round(spearman[1],3))
				
		ftext="$r^{2}$ = "+(str)(round(pearson[0]**2,3))+" "+p
		gtext=r'$\rho$'+" = "+(str)(round(spearman[0],3))+" "+spearman_p
		more_text="n= " + (str)(len(new_lifespans))
	
		plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
		plt.figtext(.15,.75,gtext,fontsize=20,ha='left')
		plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
		
	
		plt.savefig(save_name)
		plt.gcf().clf
		plt.close()	

	else:
		
		data = worms.get_time_range(feature+'_z', min_age, max_age)
		lifespans=worms.get_feature(age_feature)/24
		spans=worms.get_feature(age_feature)/24

		save_name=save_dir+'/'+feature + ' '+function+' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24)) +'_'+age_feature+'.svg'
		n_function=numpy_functions[function]
		averages=[]
		new_lifespans=[]

		for i in range(0, len(data)):
			if lifespans[i]>=max_age/24 and len(data[i])>1:
				average=n_function(data[i][:,1])
		
				if -3<average<3:
					averages.append(average)
					new_lifespans.append(spans[i])
			

		remaining_lifespans=[i-max_age/24 for i in new_lifespans]


		color_vals = colorize.scale(new_lifespans, output_max=1)
		colors = colorize.color_map(color_vals, uint8=False)
		plt.scatter(averages,remaining_lifespans,c=colors)
		plt.xlabel(function+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
		plt.ylabel("Days of life remaining",fontdict=label_fontdict)
		plt.title(function+' '+ 'P'+miRNA +'::GFP expression at '+(str)((int)(max_age/24))+' dph ' +age_feature,fontdict=title_fontdict)
		ymin, ymax=plt.ylim()
		plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


		pearson=numpy.asarray(scipy.stats.pearsonr(remaining_lifespans, averages))
		spearman=numpy.asarray(scipy.stats.spearmanr(remaining_lifespans, averages))
		(m,b) = numpy.polyfit(averages, remaining_lifespans, 1)
		yp=numpy.polyval([m,b], averages)
		mew, mewmew=zip(*sorted(zip(averages, yp)))
		mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
		plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)

	
		if pearson[1]<.00001:
			p="p<.00001"
		else:
			p="p=" + ''+ (str)(round(pearson[1],3))
		if spearman[1]<.00001:
			spearman_p="p<.00001"
		else:
			spearman_p="p=" + '' + (str)(round(spearman[1],3))
				
		ftext="$r^{2}$ = "+(str)(round(pearson[0]**2,3))+" "+p
		gtext=r'$\rho$'+" = "+(str)(round(spearman[0],3))+" "+spearman_p
		more_text="n= " + (str)(len(new_lifespans))
	
		plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
		plt.figtext(.15,.75,gtext,fontsize=20,ha='left')
		plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
		
	
		plt.savefig(save_name)
		plt.gcf().clf
		plt.close()	

def plot_all_correlations(worms, min_age, save_dir, function='Mean',age_feature='lifespan', gfp_measures=gfp_measures, max_age=0, miRNA='',control_for=None):

	ages=[]
	lifespans=worms.get_feature('lifespan')
	if max_age==0:
		max_age=(int)(numpy.round(numpy.percentile(lifespans,10))/24)
	for i in range(min_age,max_age*24+1,24):
		ages.append(i)	
		
	for key, value in gfp_measures.items():
		correlations=[]
		p_values=[]
		for j in range(0, len(ages)-1):
			correlation,p_value, save_name=get_correlations(worms,ages[j],ages[j+1],key,save_dir,function,age_feature,miRNA=miRNA,control_for=control_for)
			correlations.append(correlation)
			p_values.append(p_value)

		new_ages=[i/24 for i in ages[1::]]
		plt.scatter(new_ages,correlations,c='navy',marker='o',s=50,edgecolor='navy')
		plt.plot(new_ages,correlations, c='navy',linestyle='--')	
		p_x=[]
		p_y=[]
		for i in range(0,len(p_values)):
			if p_values[i]<.05 and correlations[i]>=.05:
				p_y.append(correlations[i]+.03)
				p_x.append(new_ages[i])
		plt.scatter(p_x,p_y,marker=(6,2,0),c='navy',s=50)

	
		plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+age_feature, y=1.05, fontdict=title_fontdict)
		plt.xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict)
		plt.ylabel('Coefficient of determination ('+"$r^{2}$"+')',fontdict=label_fontdict)
		plt.ylim(0,.6)
		

		plt.savefig(save_name)
		plt.gcf().clf	
		plt.close()

def get_correlations(worms,min_age,max_age,key,save_dir,function,age_feature,miRNA, control_for=None):

	if control_for is not None:
		data = worms.get_time_range(key+'_z', min_age, max_age)
		control_data=worms.get_time_range(control_for+'_z', min_age, max_age)
		lifespans=worms.get_feature(age_feature)
		spans=worms.get_feature(age_feature)
		save_name=save_dir+'/'+key + ' '+function+' ' +'correlations_'+age_feature+' controlled for '+control_for+'_.svg'	
		n_function=numpy_functions[function]
		averages=[]
		new_lifespans=[]
		control_averages=[]
		for i in range(0, len(data)):
			if lifespans[i]>=max_age and len(data[i])>1:
				average=n_function(data[i][:,1])
				control_average=n_function(control_data[i][:,1])
				if -3<average<3:
					averages.append(average)
					new_lifespans.append(spans[i])
					control_averages.append(control_average)

		remaining_lifespans=[i-max_age for i in new_lifespans]

		averages=numpy.asarray(averages)
		control_averages=numpy.asarray(control_averages)
		slope_x, intercept_x,rvalue_x,pvalue_x,stderror_x=scipy.stats.linregress(x=control_averages,y=averages)
		slope_y,intercept_y,rvalue_y,pvalue_y,stderror_y=scipy.stats.linregress(x=control_averages,y=remaining_lifespans)
		obsvalues_x=averages
		predvalues_x=control_averages*slope_x+intercept_x
		residuals_x=predvalues_x-obsvalues_x
		obsvalues_y=remaining_lifespans
		predvalues_y=control_averages*slope_y+intercept_y
		residuals_y=predvalues_y-obsvalues_y
		pearson=scipy.stats.pearsonr(residuals_x,residuals_y)
		spearman=scipy.stats.spearmanr(residuals_x,residuals_y)

	else:	
		lifespans=worms.get_feature('lifespan')
		spans=worms.get_feature(age_feature)
		data=worms.get_time_range(key+'_z',min_age,max_age)
		save_name=save_dir+'/'+key + ' '+function+' ' +'correlations_'+age_feature+'_.svg'	
		n_function=numpy_functions[function]
		
		averages=[]
		new_lifespans=[]

		for i in range(0, len(data)):
			if lifespans[i]>=max_age and len(data[i])>1:
				average=n_function(data[i][:,1])
				if -3<average<3:
					averages.append(average)
					new_lifespans.append(spans[i])

		pearson=numpy.asarray(scipy.stats.pearsonr(new_lifespans, averages))

	return pearson[0]**2, pearson[1], save_name


def plot_survival_curve(worms,save_dir,fill=[],rescale=False):
	
	lifespans=worms.get_feature('lifespan')/24
	max_life=lifespans.max()
	min_life=lifespans.min()
	days=numpy.arange(2,max_life+1,.25)
	
	percent_alive = []
	

	for i in days:
		count =0
		for item in lifespans:
			if item > i:
				count=count+1
		percent_alive.append((count/len(lifespans))*100)
	if rescale:
		fill=fill/days.max()
		days=days/days.max()
		
		
	color_vals = colorize.scale(days, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)
	#reverse_colors=colors[::-1]
	plt.plot(days,percent_alive,color='gray',alpha=.3)	
	plt.scatter(days,percent_alive,color=colors)
	if len(fill)>0:
		x=numpy.linspace(fill[0],fill[1],len(percent_alive))
		y1=numpy.zeros(len(percent_alive))+100
		plt.fill_between(x,y1, color='gray', alpha=.5)
	plt.xlabel("Age (days)",fontdict=label_fontdict)
	plt.ylabel("Fraction Alive (%)",fontdict=label_fontdict)
	more_text="median lifespan = " + (str)((round(numpy.median(lifespans),1))) + " days"
	even_more_text="n = "+(str)(len(lifespans))
	plt.figtext(.15, .2, more_text, fontsize=20, ha='left', weight='bold')
	plt.figtext(.15, .15, even_more_text, fontsize=20, ha='left', weight='bold')
	if rescale:
		save_name=save_dir+'/'+'survival_curve_rescaled.svg'
	else:
		save_name=save_dir+'/'+'survival_curve.svg'
	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()

#e.g. bin_name='lifespan'
def plot_trajectories(worms, bin_name, save_dir, miRNA='', nbins=4,min_age=-numpy.inf, max_age=numpy.inf,equal_count=True,
        age_feature='age', time_units='hours',gfp_measures=gfp_measures,rescale=False,ref_strain=''):

	for key,value in gfp_measures.items():

		if rescale:

			median_gfp={}
			groups=worms.group_by([w.name.split()[0] for w in worms])	
	
			for exp, wormies in groups.items():
				data = wormies.get_time_range(key,min_age,max_age)	
				median_gfp[exp] = numpy.median([d[:,1].mean() for d in data])

			for worm in worms:
				exp = worm.name.split()[0]
				worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[ref_strain]
			
			lifespans=worms.get_feature('lifespan')
			if nbins==1:
				bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}
				save_name=save_dir+'/'+ miRNA + '_'+ key+'onegroup_rescaled.svg'
			else:		
				bins = worms.bin('lifespan',nbins=nbins, equal_count=equal_count)
				save_name=save_dir+'/'+ miRNA + '_'+ key+'_'+age_feature+'rescaled.svg' 
	
			averaged_worms = worm_data.meta_worms(bins, 'scaled_gfp', age_feature=age_feature)
			figure=averaged_worms.plot_timecourse('scaled_gfp',age_feature=age_feature,min_age=min_age,time_units=time_units) 
			 	
			
	
		else:
			lifespans=worms.get_feature('lifespan')
			if nbins==1:
				bins={'['+(str)(lifespans.min())+'-'+(str)(lifespans.max())+']':worms}
				save_name=save_dir+'/'+ miRNA + '_'+ key+'_onegroup.svg'
			else:	
				bins = worms.bin(bin_name,nbins=nbins,equal_count=equal_count)
				save_name=save_dir+'/'+ miRNA + '_'+ key+ '_' + age_feature+'_'+bin_name+'.svg'	
			averaged_worms=worm_data.meta_worms(bins, key,age_feature=age_feature)
			figure=averaged_worms.plot_timecourse(key,time_units=time_units,min_age=min_age,max_age=max_age,age_feature=age_feature)
			

		cohorts=[]

		for mew, mewmew in bins.items():
			cohorts.append(len(mewmew))	
		
		ymin, ymax=plt.ylim()
		plt.ylim(ymax=ymax+ymax*.1)

		if age_feature=='ghost_age':
			plt.xlabel(time_units + ' before death', fontdict=label_fontdict)
		
		else:
			plt.xlabel(time_units+ ' post-hatch', fontdict=label_fontdict)

		plt.ylabel('P'+miRNA+'::GFP expression ('+value+')', fontdict=label_fontdict)
		plt.title('Average ' +'P'+miRNA+'::GFP expression profiles (by'+ ' ' + bin_name+')', fontdict=title_fontdict)
		text='n = ' + (str)(cohorts)
		plt.figtext(.5, .15, text, fontdict=text_fontdict)
		
		plt.savefig(save_name)
		plt.gcf().clf
		plt.close()

