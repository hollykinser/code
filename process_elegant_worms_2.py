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

gfp_measures_pharynx={
'pharynx_percentile_95':'95th percentile intensity',
'pharynx_percentile_99':'99th percentile intensity',
'pharynx_sum':'summed intensity',
'pharynx_maximum':'maximum intensity',
'pharynx_mean':'mean intensity',
'pharynx_median':'median intensity'}
gfp_measures_vulva={
'vulva_percentile_95':'95th percentile intensity',
'vulva_percentile_99':'99th percentile intensity',
'vulva_sum':'summed intensity',
'vulva_maximum':'maximum intensity',
'vulva_mean':'mean intensity',
'vulva_median':'median intensity'}
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

def plot_all_multivariable(worms,min_age,save_dir,function,gfp_measures=gfp_measures,max_age=0,miRNA='',rescale_lifespan=False,ref_strain=''):
	ages=[]
	lifespans=worms.get_feature('lifespan')
	if max_age==0:
		max_age=(int)(numpy.round(numpy.percentile(lifespans,10)))
	ages=[min_age, max_age]	
	#for i in range(min_age,max_age*24+1,24):
		#ages.append(i)
	print(ages)		
	for key, value in gfp_measures.items():
		for j in range(0, len(ages)-1):
			multivariable_regression(worms,ages[j],ages[j+1],key,save_dir,function,miRNA=miRNA,gfp_measures=gfp_measures,rescale_lifespan=rescale_lifespan,ref_strain=ref_strain)

def multivariable_regression (worms,min_age, max_age,feature,save_dir,function,miRNA='',gfp_measures=gfp_measures, ingredients=['Average','Slope'],rescale_lifespan=False,ref_strain=''):
	if rescale_lifespan:
		median_lifespan={}
		groups=worms.group_by([w.name.split()[0] for w in worms])	
	
		for exp, wormies in groups.items():
			median_lifespan[exp] = numpy.median(wormies.get_feature('lifespan'))
		for worm in worms:
			exp = worm.name.split()[0]
			worm.scaled_lifespan = (worm.lifespan / median_lifespan[exp]) * median_lifespan[ref_strain] / 24
	
		lifespans=worms.get_feature('scaled_lifespan')


	else:	
		lifespans=worms.get_feature('lifespan')/24
	data=worms.get_time_range(feature+'_z',min_age,max_age)
	control_data=worms.get_time_range('green_yellow_excitation_autofluorescence_percentile_95_z',min_age,max_age)

	n_function=numpy_functions[function]
		
	
	averages=[]
	new_lifespans=[]
	slopes=[]
	controls=[]
	control_slopes=[]


	for i in range(0, len(data)):
		if lifespans[i]>=max_age/24 and len(data[i])>1:
			average=n_function(data[i][:,1])
			if -3<average<3:
				slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(data[i])
				slopes.append(slope)
				#control=n_function(control_data[i][:,1])
				#control_slope, c_intercept, c_r_value, c_p_value, c_std_err=scipy.stats.linregress(control_data[i])
				#controls.append(control)
				averages.append(average)
				#control_slopes.append(control_slope)
				new_lifespans.append(lifespans[i])


	mewmew={'Slope': slopes, 'Average': averages,'Lifespan': new_lifespans}
	df=DataFrame(mewmew, columns=['Slope', 'Average', 'Autofluorescence','AutoSlope','Lifespan'])
	X=df[ingredients]
	Y=df['Lifespan']
	X=sm.add_constant(X)
	model=sm.OLS(Y,X).fit()
	predictions = model.predict(X)
	print_model = model.summary()
	print(print_model)
	save_name=save_dir+'/'+feature + ' ' + function + ' bivariate' +(str)(X.columns.tolist()) +' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24))
	save_name=save_name+'.svg'			
	color_vals = colorize.scale(new_lifespans, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)
	plt.scatter(predictions,new_lifespans,c=colors)
	plt.xlabel('Predicted lifespan (days)',fontdict=label_fontdict)
	plt.ylabel("Actual lifespan (days)",fontdict=label_fontdict)
	plt.title('Mean and slope of '+ 'P'+miRNA +'::GFP expression at '+(str)((int)(max_age/24))+' dph vs. lifespan',fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


	(m,b) = numpy.polyfit(predictions, new_lifespans, 1)
	yp=numpy.polyval([m,b], predictions)
	mew, mewmew=zip(*sorted(zip(predictions, yp)))
	mew, mewmew = (list(t) for t in zip(*sorted(zip(mew, mewmew))))
	plt.plot([mew[0],mew[-1]],[mewmew[0],mewmew[-1]],c='gray',alpha=.7)

	pearson=numpy.asarray(scipy.stats.pearsonr(new_lifespans, predictions))
	spearman=numpy.asarray(scipy.stats.spearmanr(new_lifespans, predictions))
	
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

def get_gfp_on_day_wrapper(first_day, last_day,key):
	def get_gfp_on_day(worm):
		return numpy.mean(getattr(worm.td,key+'_z')[(worm.td.age>first_day)&(worm.td.age>last_day)])
	return get_gfp_on_day	

def plot_by_expression(worms, key, first_day, last_day,miRNA,save_dir,nbins=5,plot_all=False):
	lifespans=worms.get_feature('lifespan')
	bins=worms.bin(get_gfp_on_day_wrapper(first_day, last_day, key), nbins=nbins, equal_count=True)
	
	if plot_all:
		averaged_worms=worm_data.meta_worms(bins, key+'_z')
	else:	
		averaged_worms=worm_data.meta_worms({'high_worms':bins[(str)(nbins-1)],'low_worms':bins['0']}, key+'_z')
	figure=averaged_worms.plot_timecourse(key+'_z', time_units='days', min_age=first_day, max_age=numpy.percentile(lifespans,10))
	plt.xlabel('Time (days)',fontdict=label_fontdict)
	plt.ylabel('Z score ('+gfp_measures[key]+')',fontdict=label_fontdict)
	plt.title('P'+miRNA +'::GFP expression consistency over time',fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=-2,ymax=2)	
	cohorts=[]
	if plot_all:
		save_name=save_dir+'/consistencyby_'+key+'_'+(str)(nbins)+'_'+(str)(first_day)+'_'+(str)(last_day)+'_allworms.svg'
	else:
		save_name=save_dir+'/consistencyby_'+key+'_'+(str)(nbins)+'_'+(str)(first_day)+'_'+(str)(last_day)+'.svg'	
	if plot_all:
		cohorts.append((str)(len(bins['0'])))
	else:	
		cohorts.append((str)(len(bins[(str)(nbins-1)])))	
	text='n = ' + (str)(cohorts)
	plt.figtext(.5, .15, text, fontdict=text_fontdict)
	plt.plot(worms[-1].td.age[(worms[-1].td.age>first_day)&(worms[-1].td.age<numpy.percentile(lifespans,10))]/24,numpy.zeros(len(worms[-1].td.age[(worms[-1].td.age>first_day)&(worms[-1].td.age<numpy.percentile(lifespans,10))])),color='gray')	

	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()

def percentile_wrapper(lifespans, threshold1, threshold2):
	def threshold_worms_by_lifespan(worm):
		low,high=numpy.percentile(lifespans,[threshold1, threshold2])
		return low<worm.lifespan<=high
	return threshold_worms_by_lifespan	

def plot_consistency(worms, key, high_thresholds, low_thresholds,miRNA, save_dir,smooth_param=6, single_trace=False):
	lifespans=worms.get_feature('lifespan')
	high_worms=worms.filter(percentile_wrapper(lifespans, high_thresholds[0],high_thresholds[1]))
	low_worms=worms.filter(percentile_wrapper(lifespans,low_thresholds[0],low_thresholds[1]))
	high_sums=[]

	if single_trace:

		averaged_worms=worm_data.meta_worms({'high_worms':high_worms,'low_worms':low_worms}, key+'_z')
		figure=averaged_worms.plot_timecourse(key+'_z',time_units='days',min_age=3*24,max_age=numpy.percentile(lifespans,10))
		save_name=save_name=save_dir+'/'+key+'_'+(str)(low_thresholds[0])+'_'+(str)(high_thresholds[0])+'_'+'consistency_singletrace.svg' 
	else:	
		save_name=save_dir+'/'+key+'_'+(str)(low_thresholds[0])+'_'+(str)(high_thresholds[0])+'_'+'consistency_allworms.svg' 
		for worm in high_worms:

			plt.plot(worm.td.age[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))]/24,smooth(getattr(worm.td,key+'_z')[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))],smooth_param), color='goldenrod',alpha=.5)
	  	
		for worm in low_worms:
			plt.plot(worm.td.age[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))]/24,smooth(getattr(worm.td,key+'_z')[(worm.td.age>3*24)&(worm.td.age<numpy.percentile(lifespans,10))],smooth_param), color='indigo',alpha=.5)
	plt.plot(high_worms[0].td.age[(high_worms[0].td.age>3*24)&(high_worms[0].td.age<numpy.percentile(lifespans,10))]/24,numpy.zeros(len(high_worms[0].td.age[(high_worms[0].td.age>3*24)&(high_worms[0].td.age<numpy.percentile(lifespans,10))])),color='gray')

			
	plt.xlabel('Time (days)',fontdict=label_fontdict)
	plt.ylabel('Z score ('+gfp_measures[key]+')',fontdict=label_fontdict)
	plt.title('P'+miRNA +'::GFP expression consistency over time',fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=-3,ymax=3)	
	text='n = ' + (str)(len(high_worms))+ '    '+(str)(len(low_worms))
	plt.figtext(.5, .15, text, fontdict=text_fontdict)
	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()

#make sure prefix is 'mew ' (space is important) otherwise each worm will be z_scored on itself and your results will make no sense
def z_scoring(worms, age_feature,gfp_measures=gfp_measures):
	groups=worms.group_by([w.name.split()[0] for w in worms])
	for exp, wormies in groups.items():
		for feature in gfp_measures:
			wormies.z_transform(feature,age_feature=age_feature)



def smooth(y, box_pts):
	box=numpy.ones(box_pts)/box_pts
	y_smooth=numpy.convolve(y,box,mode='same')
	return y_smooth


def process_worms(paths,prefixes=['mew ']):
	wormies=[]
	for path, prefix in zip(paths,prefixes):
		worms=worm_data.read_worms(path+'/*.tsv',name_prefix=prefix)
		wormies=wormies+worms
	
	return wormies



def plot_z_scores(worms, feature, min_age, max_age=0,smooth_param=1,perc=[]):
	worms.z_transform(feature)
	lifespans=worms.get_feature('lifespan')
	if max_age==0:
		max_age=(int)(numpy.round(numpy.percentile(lifespans,10))/24)
	z_scores=worms.get_time_range(feature+'_z',min_age=min_age,max_age=max_age)	
	if len(perc)>0:
		worms=worms.filter(percentile_wrapper(perc[0],perc[1],worms))	
	
	for i in range(0, len()):
		plt.plot(z_scores[i][:,0],smooth(z_scores[i][:,1],smooth_param))
	plt.plot(z_scores[0][:,0],numpy.zeros(len(z_scores[0])))
	plt.show()


def plot_all_correlations_ghost(worms, save_dir, function, min_ghost_age,age_feature,gfp_measures=gfp_measures, miRNA=''):
	ages=[]
	lifespans=worms.get_feature('adultspan')
	max_ghost_age=-(int)(numpy.round(numpy.percentile(lifespans,10)/24))
	for i in range(max_ghost_age*24+24,min_ghost_age+24,24):
		ages.append(i)
		
	for key, value in gfp_measures.items():
		correlations=[]
		p_values=[]
		for j in range(0, len(ages)-1):
			correlation,p_value, save_name=get_correlations_ghost(worms,ages[j],ages[j+1],key,save_dir,function,age_feature,miRNA=miRNA)
			
			correlations.append(correlation)
			p_values.append(p_value)
		new_ages=[i/24 for i in ages[0:-1]]
		plt.scatter(new_ages,correlations,c='navy',marker='o',s=50,edgecolor='navy')
		plt.plot(new_ages,correlations, c='navy',linestyle='--')	
		p_x=[]
		p_y=[]
		for i in range(0,len(p_values)):
			if p_values[i]<.05 and correlations[i]>=.05:
				p_y.append(correlations[i]+.03)
				p_x.append(new_ages[i])

		plt.scatter(p_x,p_y,marker=(6,2,0),c='navy',s=50)
		plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+ age_feature+' over time', y=1.05, fontdict=title_fontdict)
		plt.xlabel('Timepoint (day before death)', fontdict=label_fontdict)
		plt.ylabel('Coefficient of determination ('+"$r^{2}$"+')',fontdict=label_fontdict)
		plt.ylim(0,1)
		

		plt.savefig(save_name)
		plt.gcf().clf	
		plt.close()

def get_correlations_ghost(worms,min_age,max_age,key,save_dir,function,age_feature,miRNA):	
	
	lifespans=worms.get_feature(age_feature)/24
	data=worms.get_time_range(key+'_z',min_age,max_age,age_feature='ghost_age')
	save_name=save_dir+'/'+key + ' '+function+' ' +'ghost correlations '+age_feature+'_'

	
	save_name=save_name+'.svg'	
	n_function=numpy_functions[function]
		
	
	averages=[]
	new_lifespans=[]

	for i in range(0, len(data)):
		if lifespans[i]>=abs(min_age/24)and len(data[i])>1:
			average=n_function(data[i][:,1])
			if -3<average<3:
				averages.append(average)
				new_lifespans.append(lifespans[i])

	pearson=numpy.asarray(scipy.stats.pearsonr(new_lifespans, averages))

	return pearson[0]**2, pearson[1], save_name

def timepoint_of_decline(worm):
	median_movement=numpy.median(worm.td.centroid_dist[worm.td.stage=='adult'])
	smoothed=smooth(worm.td.centroid_dist,6)
	timepoints=worm.td.age[smoothed>median_movement*.7]
	return timepoints[timepoints>3*24]

def plot_movement_and_lifespans(worms, save_dir):
	
	lifespans=worms.get_feature('lifespan')/24
	save_name=save_dir+'/movementvslifespan.svg'	

	decline_points=[]
	for worm in worms:
		timepoints=timepoint_of_decline(worm)
		decline_points.append(timepoints[-1]/24)	


	color_vals = colorize.scale(lifespans, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)
	plt.scatter(decline_points,lifespans,c=colors)
	plt.xlabel('Movementspan (days)',fontdict=label_fontdict)
	plt.ylabel('Lifespan (days)',fontdict=label_fontdict)
	plt.title('Lifespan vs. movementspan',fontdict=title_fontdict)

	pearson=numpy.asarray(scipy.stats.pearsonr(decline_points,lifespans))
	spearman=numpy.asarray(scipy.stats.spearmanr(decline_points,lifespans))
	(m,b) = numpy.polyfit(decline_points,lifespans, 1)
	yp=numpy.polyval([m,b], lifespans)
	mew, mewmew=zip(*sorted(zip(lifespans, yp)))
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
	more_text="n= " + (str)(len(decline_points))
	
	plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
	plt.figtext(.15,.75,gtext,fontsize=20,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
		
	
	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()			
	

	

def get_correlations_movement(worms,min_age,max_age,key,save_dir,function,age_feature,miRNA):

	lifespans=worms.get_feature('lifespan')
	data=worms.get_time_range(key+'_z',min_age,max_age)
	save_name=save_dir+'/'+key + ' '+function+' ' +'correlations_movement.svg'	
	n_function=numpy_functions[function]

	decline_points=[]
	for worm in worms:
		timepoints=timepoint_of_decline(worm)
		decline_points.append(timepoints[-1]/24)	
		
	averages=[]
	new_lifespans=[]
	new_decline_points=[]

	for i in range(0, len(data)):
		if lifespans[i]>=max_age and len(data[i])>1:
			average=n_function(data[i][:,1])
			if -3<average<3:
				averages.append(average)
				new_decline_points.append(decline_points[i])

	pearson=numpy.asarray(scipy.stats.pearsonr(new_decline_points, averages))

	return pearson[0]**2, pearson[1], save_name	

def plot_scatter_movement(worms, min_age, max_age,feature,save_dir,age_feature,function,miRNA='',gfp_measures=gfp_measures):
		
	data = worms.get_time_range(feature+'_z', min_age, max_age)
	lifespans=worms.get_feature(age_feature)/24
	spans=worms.get_feature(age_feature)/24
	save_name=save_dir+'/'+feature + ' '+function+' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24)) +'_movement.svg'
	n_function=numpy_functions[function]

	decline_points=[]
	for worm in worms:
		timepoints=timepoint_of_decline(worm)
		decline_points.append(timepoints[-1]/24)	
	
	averages=[]
	new_decline_points=[]

	for i in range(0, len(data)):
		if lifespans[i]>=max_age/24 and len(data[i])>1:
			average=n_function(data[i][:,1])
			if -3<average<3:
				averages.append(average)
				new_decline_points.append(decline_points[i])
	
	color_vals = colorize.scale(new_decline_points, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)
	plt.scatter(averages,new_decline_points,c=colors)
	plt.xlabel(function+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
	plt.ylabel("Movementspan (days)",fontdict=label_fontdict)
	plt.title(function+' '+ 'P'+miRNA +'::GFP expression at '+(str)((int)(max_age/24))+' dph vs. movementspan',fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


	pearson=numpy.asarray(scipy.stats.pearsonr(new_decline_points, averages))
	spearman=numpy.asarray(scipy.stats.spearmanr(new_decline_points, averages))
	(m,b) = numpy.polyfit(averages, new_decline_points, 1)
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
	more_text="n= " + (str)(len(new_decline_points))
	
	plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
	plt.figtext(.15,.75,gtext,fontsize=20,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
		
	
	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()			

def plot_all_scatters(worms,min_age,save_dir,age_feature,function,gfp_measures=gfp_measures,max_age=0,miRNA='',movement=False, auto=False,control_for=None):
	
	ages=[]
	lifespans=worms.get_feature('lifespan')

	if max_age==0:
		max_age=(int)(numpy.round(numpy.percentile(lifespans,10)/24))
	for i in range(min_age,max_age*24+1,24):
		ages.append(i)
		print(ages)

	for key, value in gfp_measures.items():
		for j in range(0, len(ages)-1):
			if movement:
				plot_scatter_movement(worms,ages[j],ages[j+1],key,save_dir,age_feature,function,miRNA=miRNA,gfp_measures=gfp_measures)
			if auto:
				plot_scatter_auto(worms, ages[j], ages[j+1], key, save_dir, function, miRNA=miRNA,gfp_measures=gfp_measures)	
			else:
				plot_scatter(worms,ages[j],ages[j+1],key,save_dir,age_feature,function,miRNA=miRNA,gfp_measures=gfp_measures,control_for=control_for)

def plot_scatter_auto(worms,min_age, max_age, feature, save_dir,function, miRNA='',gfp_measures=gfp_measures):
	data = worms.get_time_range(feature+'_z', min_age, max_age)
	lifespans=worms.get_feature('lifespan')
	auto_data=worms.get_time_range('vulva_percentile_99_z',min_age, max_age)


	save_name=save_dir+'/'+feature + ' '+function+' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24)) +'_auto.svg'
	n_function=numpy_functions[function]
	averages=[]
	auto_averages=[]
	for i in range(0, len(data)):
		if lifespans[i]>=max_age/24 and len(data[i])>1:
			average=n_function(data[i][:,1])
			auto_average=n_function(auto_data[i][:,1])
			if -3<average<3 and -3<auto_average<3:
				averages.append(average)
				auto_averages.append(auto_average)
	

	color_vals = colorize.scale(averages, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)

	plt.scatter(averages,auto_averages,c=colors)
	plt.xlabel(function+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
	plt.ylabel("Average autofluorescence (95th percentile intensity)",fontdict=label_fontdict)
	plt.title(function+' '+ 'P'+miRNA +'::GFP expression vs. autofluorescence at '+(str)((int)(max_age/24))+' dph',fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


	pearson=numpy.asarray(scipy.stats.pearsonr(auto_averages, averages))
	spearman=numpy.asarray(scipy.stats.spearmanr(auto_averages, averages))
	(m,b) = numpy.polyfit(averages, auto_averages, 1)
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
	more_text="n= " + (str)(len(averages))
	
	plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
	plt.figtext(.15,.75,gtext,fontsize=20,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
		
	
	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()	

def plot_scatter(worms, min_age, max_age,feature,save_dir,age_feature,function,miRNA='',gfp_measures=gfp_measures,control_for=None):
	if control_for is not None:
		data = worms.get_time_range(feature, min_age, max_age)
		control_data=worms.get_time_range(control_for, min_age, max_age)
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
				#if -3<average<3:
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
		
		data = worms.get_time_range(feature, min_age, max_age)
		lifespans=worms.get_feature(age_feature)/24
		spans=worms.get_feature(age_feature)/24

		save_name=save_dir+'/'+feature + ' '+function+' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24)) +'_'+age_feature+'.svg'
		n_function=numpy_functions[function]
		averages=[]
		new_lifespans=[]

		for i in range(0, len(data)):
			if lifespans[i]>=max_age/24 and len(data[i])>1:
				average=n_function(data[i][:,1])
		
				#if -3<average<3:
				averages.append(average)
				new_lifespans.append(spans[i])
			




		color_vals = colorize.scale(new_lifespans, output_max=1)
		colors = colorize.color_map(color_vals, uint8=False)
		plt.scatter(averages,new_lifespans,c=colors)
		plt.xlabel(function+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
		plt.ylabel("Days of life remaining",fontdict=label_fontdict)
		plt.title(function+' '+ 'P'+miRNA +'::GFP expression at '+(str)((int)(max_age/24))+' dph ' +age_feature,fontdict=title_fontdict)
		ymin, ymax=plt.ylim()
		plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


		pearson=numpy.asarray(scipy.stats.pearsonr(new_lifespans, averages))
		spearman=numpy.asarray(scipy.stats.spearmanr(new_lifespans, averages))
		(m,b) = numpy.polyfit(averages, new_lifespans, 1)
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

def plot_all_correlations(worms, min_age, save_dir, function,age_feature, gfp_measures=gfp_measures, max_age=0, miRNA='',movement=False,control_for=None):

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
			if movement:
				correlation,p_value, save_name=get_correlations_movement(worms,ages[j],ages[j+1],key,save_dir,function,age_feature,miRNA=miRNA)

			else:	
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
		if movement:
			plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with movementspan', y=1.05, fontdict=title_fontdict)
		else:	
			plt.title('Correlation of '+miRNA + ' expression ' + '('+ value+')'+ ' with '+age_feature, y=1.05, fontdict=title_fontdict)
		plt.xlabel('Timepoint (day post-hatch)', fontdict=label_fontdict)
		plt.ylabel('Coefficient of determination ('+"$r^{2}$"+')',fontdict=label_fontdict)
		plt.ylim(0,.6)
		

		plt.savefig(save_name)
		plt.gcf().clf	
		plt.close()

def get_correlations(worms,min_age,max_age,key,save_dir,function,age_feature,miRNA, control_for=None):

	if control_for is not None:
		data = worms.get_time_range(key, min_age, max_age)
		control_data=worms.get_time_range(control_for, min_age, max_age)
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
				#if -3<average<3:
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
		data=worms.get_time_range(key,min_age,max_age)
		save_name=save_dir+'/'+key + ' '+function+' ' +'correlations_'+age_feature+'_.svg'	
		n_function=numpy_functions[function]
		
		averages=[]
		new_lifespans=[]

		for i in range(0, len(data)):
			if lifespans[i]>=max_age and len(data[i])>1:
				average=n_function(data[i][:,1])
				#if -3<average<3:
				averages.append(average)
				new_lifespans.append(spans[i])

		pearson=numpy.asarray(scipy.stats.pearsonr(new_lifespans, averages))

	return pearson[0]**2, pearson[1], save_name


def plot_all_scatters_ghost(worms,save_dir,function,age_feature,min_ghost_age=-24,gfp_measures=gfp_measures,miRNA=''):
	ages=[]
	lifespans=worms.get_feature('adultspan')

	
	max_ghost_age=-(int)(numpy.round(numpy.percentile(lifespans,10)/24))
	for i in range(max_ghost_age*24+24,min_ghost_age+24,24):
		ages.append(i)
	print(ages)

	for key, value in gfp_measures.items():
		for j in range(0, len(ages)-1):
			plot_scatter_ghost(worms,ages[j],ages[j+1],key,save_dir,function,age_feature,miRNA=miRNA,gfp_measures=gfp_measures)	

def plot_scatter_ghost(worms, min_age, max_age,feature,save_dir,function,age_feature,miRNA='',gfp_measures=gfp_measures):
	
	lifespans=worms.get_feature(age_feature)/24
	data=worms.get_time_range(feature+'_z',min_age,max_age,age_feature='ghost_age')
	save_name=save_dir+'/'+feature + ' '+function+' ' +(str)((int)(min_age/24)) + ' ' + (str)((int)(max_age/24)) +'_'+age_feature
	n_function=numpy_functions[function]
	averages=[]
	new_lifespans=[]
	for i in range(0, len(data)):
		if lifespans[i]>=abs(min_age/24) and len(data[i])>1:
			average=n_function(data[i][:,1])
			if -3<average<3:
				averages.append(average)
				new_lifespans.append(lifespans[i])
	save_name=save_name+'.svg'			
	color_vals = colorize.scale(new_lifespans, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)
	plt.scatter(averages,new_lifespans,c=colors)
	plt.xlabel(function+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
	plt.ylabel(age_feature +'(days)',fontdict=label_fontdict)
	plt.title(function+' '+ 'P'+miRNA +'::GFP expression at '+(str)(abs((int)(min_age/24)))+' day before death vs. '+age_feature,fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


	pearson=numpy.asarray(scipy.stats.pearsonr(new_lifespans, averages))
	spearman=numpy.asarray(scipy.stats.spearmanr(new_lifespans, averages))
	(m,b) = numpy.polyfit(averages, new_lifespans, 1)
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


def peak_gfp(worm,feature,age_max,age_min):
	ages=(worm.td.age<age_max)&(worm.td.age>age_min)
	data=getattr(worm.td,feature)
	data=data[ages]
	return worm.td.age[ages][data.argmax()]

def plot_peaks(worms,feature,miRNA,save_dir,age_max=10*24,age_min=2*24):
	adultspans=worms.get_feature('lifespan')
	peaks=[]
	new_adultspans=[]
	for worm in worms:
		peaks.append(peak_gfp(worm,feature,age_max,age_min)/24)
	
	
	remaining=adultspans-peaks
	
	save_name=save_dir+'/peak'+feature+'.svg'			
	color_vals = colorize.scale(remaining, output_max=1)
	colors = colorize.color_map(color_vals, uint8=False)
	plt.scatter(peaks,remaining/24,c=colors)
	plt.xlabel('Time of peak '+ ' '+'P'+miRNA+'::GFP expression (' +gfp_measures[feature]+')',fontdict=label_fontdict)
	plt.ylabel('Days of life remaining',fontdict=label_fontdict)
	plt.title('Time of peak '+' '+ 'P'+miRNA +'::GFP expression vs. days of life remaining',fontdict=title_fontdict)
	ymin, ymax=plt.ylim()
	plt.ylim(ymin=ymin-ymin*.05,ymax=ymax+ymax*.1)


	pearson=numpy.asarray(scipy.stats.pearsonr(peaks, remaining/24))
	spearman=numpy.asarray(scipy.stats.spearmanr(peaks, remaining/24))
	(m,b) = numpy.polyfit(peaks,remaining/24, 1)
	yp=numpy.polyval([m,b], peaks)
	mew, mewmew=zip(*sorted(zip(peaks, yp)))
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
	more_text="n= " + (str)(len(remaining))
	
	plt.figtext(.15,.8,ftext,fontsize=20,ha='left')
	plt.figtext(.15,.75,gtext,fontsize=20,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=20, ha='left')
		
	
	plt.savefig(save_name)
	plt.gcf().clf
	plt.close()				

def plot_one_group(worms, save_dir,miRNA='', min_age=-numpy.inf, max_age=numpy.inf,
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
				worm.td.scaled_gfp = getattr(worm.td,key)/ median_gfp[exp] * median_gfp[ref_strain]/24
			
		
			save_name=save_dir+'/'+ miRNA + '_'+ key+'_movingmean_rescaled.svg' 
	
			trend_x, mean_trend,std_trend=worms.z_transform('scaled_gfp')
			 	
			
	
		else:
			save_name=save_dir+'/'+ miRNA + '_'+ key+'_movingmean.svg' 
	
			trend_x, mean_trend,std_trend=worms.z_transform(key)
			 	
		listy=(list)(trend_x)
		listy=[round(i) for i in listy]	
	
		plt.plot(trend_x[listy.index(min_age):], mean_trend[listy.index(min_age):])
		
	
		
		plt.xlabel(time_units+ ' post-hatch', fontdict=label_fontdict)

		plt.ylabel('P'+miRNA+'::GFP expression ('+value+')', fontdict=label_fontdict)
		plt.title('Average ' +'P'+miRNA+'::GFP expression profiles', fontdict=title_fontdict)
		text='n = ' + (str)(len(worms))
		plt.figtext(.5, .15, text, fontdict=text_fontdict)
		
		plt.savefig(save_name)
		plt.gcf().clf
		plt.close()
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

