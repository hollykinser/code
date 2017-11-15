
from pylab import *
import numpy
import scipy.stats
import matplotlib.patches as mpatches


import wormPhysiology.analyzeHealth.selectData as sd
import wormPhysiology.basicOperations.folderStuff as fs
import wormPhysiology.analyzeHealth.characterizeTrajectories_health as ct
import wormPhysiology.analyzeHealth.computeStatistics as cs


terms={
		'expressionarea_gfp': 'expression area', 
		'expressionareafraction_gfp': 'expression area fraction', 
		'expressionmean_gfp': 'expression area mean', 
		'highexpressionarea_gfp': 'high expression area', 
		'highexpressionareafraction_gfp': 'high expression area fraction', 
		'highexpressionintegrated_gfp': 'high expression area integrated', 
		'highexpressionmean_gfp': 'high expression area mean', 
		'integrated99_gfp':'summed 99th percentile intensity',
		'integrated_gfp': 'integrated intensity', 
		'intensity_70': '70th percentile intensity', 
		'intensity_80': '80th percentile intensity', 
		'intensity_90': '90th percentile intensity', 
		'intensity_95': '95th percentile intensity',
		'max_gfp': 'maximum intensity',
		'median_gfp': 'median intensity', 
		'percentile95_gfp': '95th percentile intensity', 
		'percentile99_gfp': '99th percentile intensity',
		'percentile99area_gfp':'99th percentile area',
		'total_size': 'total size',
		'health': 'health',
		'movement': 'movement',
		'autofluorescence': 'autofluorescence',
		'size': 'size'} 

def master_function(measures, data_dirs, save_dir, first_day, last_day, miRNA, fluor_measures=False, auto_measures=False, size_measures=False,health_measures=False,control_for=None):
	
	#finds average values for each measure on each day b/t first and last day
	directory_bolus=organize(data_dirs)
	complete_worm_df=ct.CompleteWormDF(directory_bolus,save_dir) 
	all_means=[]
	for i in range(0, len(measures)):
		means, lifespans, worms,abs_healthspans,frac_healthspans=get_data(complete_worm_df, measures[i], first_day, last_day)
		all_means.append(means)
	#can supply one measure to control correlations for	
	if control_for:
		control_means, lifespans,worms=get_data(data_dirs,save_dir,control_for,first_day,last_day)	
	else:
		control_means=None

	days=[]
	first_day=(int)((float)(first_day))
	last_day=(int)((float)(last_day))
	megalist=[]

	for k in range(first_day, last_day+1,1):
		days.append(k)
	for i in range(0, len(all_means)):
		correlation_list=[]
		for j in range(0, len(days)):
			if fluor_measures:
				if health_measures==False: 
					title="Mean "+miRNA+ " expression at "+ (str)(days[j]) + " dph vs. lifespan"
					y_label="Mean "+miRNA + " expression ( "+terms[measures[i]]+ " )"
					save_name= "Mean "+miRNA + " expression "+ (str)(days[j])+ " dph " + measures[i]
					correlation=run_stats(all_means[i][j], lifespans, title, y_label,"Lifespan (days)",  save_dir, save_name)	
				else:
					title="Mean "+miRNA+ " expression at "+ (str)(days[j]) + " dph vs. absolute healthspan"
					y_label="Mean "+miRNA + " expression ( "+terms[measures[i]]+ " )"
					save_name= "Mean "+miRNA + " expression "+ (str)(days[j])+ " dph " + measures[i] + " _abshealth"
					correlation=run_stats(all_means[i][j], abs_healthspans, title, y_label,"Absolute healthspan",  save_dir, save_name)	
			if auto_measures:
				title="Mean autofluorescence at "+ (str)(days[j]) + " dph vs. lifespan"
				y_label="Mean autofluorescence ( "+measures[i]+ " )"
				save_name= "Mean autofluorescence "+(str)(days[j])+ " dph " + measures[i] 
				correlation=run_stats(all_means[i][j], lifespans, title, y_label,"Lifespan (days)",  save_dir, save_name)
			if size_measures:
				title="Total size at "+ (str)(days[j]) + " dph vs. lifespan"
				y_label="Total size (pixels)"
				save_name= "Total size  "+(str)(days[j])+ " dph " 
				correlation=run_stats(all_means[i][j], lifespans, title, y_label,"Lifespan (days)",  save_dir, save_name)
			if fluor_measures==False and auto_measures==False and size_measures==False:
				title="Mean "+measures[i]+" at "+(str)(days[j]) + " dph vs. lifespan"
				y_label="Mean " +measures[i]
				save_name="Mean "+measures[i]+ (str)(days[j])+ " dph "
				correlation=run_stats(all_means[i][j], lifespans, title, y_label,"Lifespan (days)",  save_dir, save_name)
			if control_for:
				save_name=save_name+" controlled for "+(str)(control_for)				
				correlation=run_stats(all_means[i][j], lifespans, title, y_label,"Lifespan (days)",  save_dir, save_name,control=control_means[j])
			
			correlation_list.append(correlation**2)
		megalist.append(correlation_list)	
		make_correlation_plots(correlation_list,days,measures[i], miRNA, save_dir)
	return megalist,days,all_means	

def cohort_values_over_time(complete_worm_df, measure, out_dir, miRNA):
	ages=complete_worm_df.ages
	life_cohorts,bin_lifes,my_bins,my_colors=sd.life_cohort_bins(complete_worm_df,bin_width_days=2)
	averages=[]
	plt.style.use('seaborn-white')
	plt.xlabel('Days post-hatch', fontdict={'size':12,'family':'calibri'})

	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	plt.title('Average '+miRNA +' expression profiles of different lifespan cohorts',y=1.05,fontdict={'size':14,'family':'calibri'})
	plt.ylabel(miRNA+ ' expression ('+ terms[measure]+')', fontdict={'size':12,'family':'calibri'})
	more_text=''
	for i in range(0, len(life_cohorts)):
		more_text=more_text+'       '+(str)(len(life_cohorts[i]))

	plt.figtext(.5, .2, more_text, fontsize=12, ha='left',family='calibri')
	for i in range(0, len(life_cohorts)):
		averages.append(sd.cohort_trace(complete_worm_df,life_cohorts[i],measure,False))
	for i in range(0, len(life_cohorts)):
		plt.plot(ages, averages[i], color=my_colors[i])	
	save_name='averages'+measure+'profiles'	

	plt.gcf()
	plt.savefig(out_dir+'/'+save_name+'.png')
	plt.show()
	plt.gcf().clf		  	 


def make_aggregate_plot(megalist,days, miRNA,measures_used,measures_wanted):
	indices=[]
	colors=['mediumseagreen', 'deepskyblue', 'gray', 'aqua', 'mediumspringgreen', 'goldenrod', 'orange','navy','black', 'magenta', 'purple']
	for measure in measures_wanted:
		indices.append(measures_used.index(measure))

	#plt.plot(days,megalist[0], 'mediumseagreen', days, megalist[1], 'deepskyblue', days, megalist[2], 'gray', days, megalist[3], 'aqua', days, megalist[4], 'mediumspringgreen', days, megalist[5], 'goldenrod', days, megalist[6], 'navy', days, megalist[7], 'black', days, megalist[8], 'magenta', days, megalist[9], 'purple')
	patches=[]
	for indexa, indexb in enumerate(indices):
		patches.append(mpatches.Patch(color=colors[indexa], label=terms[measures_used[indexb]]))
		plt.plot(days, megalist[indexb], colors[indexa])


	

	mediumseagreen_patch=mpatches.Patch(color='mediumseagreen', label=terms[measures_used[0]])
	deepskyblue_patch=mpatches.Patch(color='deepskyblue', label=terms[measures_used[1]])
	gray_patch=mpatches.Patch(color='gray', label=terms[measures_used[2]])
	aqua_patch=mpatches.Patch(color='aqua', label=terms[measures_used[3]])
	mediumspringgreen_patch=mpatches.Patch(color='mediumspringgreen', label=terms[measures_used[4]])
	goldenrod_patch=mpatches.Patch(color='goldenrod', label=terms[measures_used[5]])
	orange_patch=mpatches.Patch(color='orange', label=terms[measures_used[6]])
	navy_patch=mpatches.Patch(color='navy', label=terms[measures_used[7]])
	black_patch=mpatches.Patch(color='black', label=terms[measures_used[8]])
	magenta_patch=mpatches.Patch(color='magenta', label=terms[measures_used[9]])
	
	

	#many_patches=[mediumseagreen_patch,deepskyblue_patch,gray_patch,aqua_patch,mediumspringgreen_patch,goldenrod_patch,navy_patch,black_patch,magenta_patch]
	
	#less_patches=[]
	#for index in indices:
		#less_patches.append(many_patches[index])
	plt.legend(handles=patches, loc='upper right')
	plt.title('Correlation of '+miRNA +' expression with fractional healthspan vs. timepoint measured', y=1.05,fontdict={'size':14, 'family':'calibri'})
	ymin, ymax = plt.ylim()
	plt.ylim(ymin,ymax+.02)
	plt.xlabel('Timepoint (dph)', fontdict={'size':12,'family':'calibri'})
	plt.ylabel('Coefficient of determination (r^2)', fontdict={'size':12,'family':'calibri'})
	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	plt.show()

def make_correlation_plots(correlation_list, days,measure, miRNA, save_dir):
	plt.style.use('seaborn-white')
	plt.plot(days,correlation_list, c='mediumseagreen')
	plt.title('Correlation of '+miRNA + ' expression ' + '('+ terms[measure]+')'+ ' with lifespan vs. timepoint', y=1.05, fontdict={'size':14,'family':'calibri'})
	plt.xlabel('Timepoint (dph)', fontdict={'size':12,'family':'calibri'})
	plt.ylabel('Coefficient of determination (r^2)',fontdict={'size':12,'family':'calibri'})
	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	ymin, ymax = plt.ylim()
	plt.ylim(ymin,ymax+.02)
	plt.gcf()
	plt.savefig(save_dir+'/'+measure+'correlationtrajectories.png')
	plt.show()
	plt.gcf().clf

	
	

def organize(data_dirs):
	work_dir='/Volumes/purplearray/Kinser_Holly/work_dir' 
	human_dir='/Volumes/purplearray/Kinser_Holly/utilities'
	extra_dirs=[]
	experiment_dirs=[]
	annotations_dirs=[]
	for i in range(0, len(data_dirs)):
		extra_dirs.append(None)
		experiment_dirs.append(None)
		annotations_dirs.append(None)
	directory_bolus=fs.DirectoryBolus(work_dir, human_dir, data_dirs,extra_dirs, experiment_dirs, annotations_dirs, done=len(data_dirs), ready=len(data_dirs))
	return directory_bolus



#first_day: first day post hatch you want to look at (i.e. '3.0')
#last_day: last day post hatch you want to look at (i.e. '8.0')

def get_data(complete_worm_df,select_measure,first_day,last_day):

	#directory_bolus=organize(data_dirs)
	#complete_worm_df=ct.CompleteWormDF(directory_bolus,save_dir)   
	#selects timepoints at 3dph onward    
	#made change 7/27/17, +8 instead of +1 (was only capturing data at t=7.0 instead of 7.0-7.7)
	select_times=complete_worm_df.times[complete_worm_df.times.index(first_day):(complete_worm_df.times.index(last_day)+8)]
	#measures=complete_worm_df.measures
	#select_measures=['median_gfp','percentile95_gfp','integrated_gfp','expressionarea_gfp','expressionmean_gfp','highexpressionarea_gfp','highexpressionintegrated_gfp','highexpressionmean_gfp','expressionareafraction_gfp','total_size','intensity_70','intensity_80','intensity_90','intensity_95']
	lifespans=sd.get_lifespans(complete_worm_df)/24
	worms=complete_worm_df.worms
	data=complete_worm_df.mloc(worms,[select_measure],select_times)
	abs_healthspans=cs.get_spans(complete_worm_df,'health', method='overall_time')/24
	frac_healthspans=abs_healthspans/lifespans
	#index=measures.index(select_measure)
	last_day=(float)(last_day)
	last_day=(int)(last_day)
	first_day=(float)(first_day)
	first_day=(int)(first_day)
	datakins=[]
	for i in range(0, (last_day-first_day)+1):
		datakins.append((data[:,0,i*8:((i*8)+8)]))
		#datakins=[data[:,index,0:7],data[:,index,8:15],data[:,index,16:23],data[:,index,24:31],data[:,index,32:39]]#,data[:,index,40:47]]

	means=[]
	for i in range(0,len(datakins)):
		means.append([])
	for i in range(0, len(datakins)):
		for j in range(0,len(datakins[i])):
			means[i].append(numpy.mean(datakins[i][j]))

	nan_positions=[]        
	for j in range(0,len(means[0])):
		if numpy.isnan(means[len(means)-1][j]):
				nan_positions.append(j)
	for i in range(0, len(means)):	
		mew=numpy.delete(means[i],nan_positions)
		means[i]=mew
	lifespans=numpy.delete(lifespans,nan_positions)
	abs_healthspans=numpy.delete(abs_healthspans,nan_positions)
	frac_healthspans=numpy.delete(frac_healthspans,nan_positions)
	worms=numpy.delete(worms,nan_positions)
	print(nan_positions)
	return means, lifespans,worms,abs_healthspans,frac_healthspans

def make_survival_curves(lifespans, title, x_label,y_label):
	
	total_worms=len(lifespans)

	max=lifespans.max()
	min=lifespans.min()

	days=numpy.arange(min-1,max+1,.5)
	
	percent_alive = []
	

	for i in days:
		sum =0
		for item in lifespans:
			if item > i:
				sum=sum+1
		percent_alive.append((sum/total_worms)*100)		
	plt.style.use('seaborn-white')
	plot(days,percent_alive, color='goldenrod')
	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	plt.xlim(xmin=(min-3))
	plt.title(title,y=1.05,fontdict={'size':14,'family':'calibri'})
	plt.xlabel(x_label,fontdict={'size':12,'family':'calibri'})
	plt.ylabel(y_label,fontdict={'size':12,'family':'calibri'})
	ftext="mean lifespan="+(str)(lifespans.mean()) + " days"
	even_more_text="median lifespan = " + (str)(numpy.percentile(lifespans,50)) + " days"
	more_text="n= " + (str)(lifespans.size)
	plt.figtext(.5,.8,ftext,fontsize=11,ha='left',family='calibri')
	plt.figtext(.8, .2, more_text, fontsize=11, ha='left', family='calibri')
	plt.figtext(.5, .75, even_more_text, fontsize=11, ha='left',family='calibri')
	plt.gcf()
	#plt.ion()
	#plt.savefig(out_dir+'/'+title+'.png')
	plt.show()
	plt.gcf().clf

def measure_slopes(measure, complete_worm_df, first_day, last_day):
	worms=complete_worm_df.worms
	lifespans=sd.get_lifespans(complete_worm_df)/24
	select_times=complete_worm_df.times[complete_worm_df.times.index(first_day):(complete_worm_df.times.index(last_day)+1)]
	data=complete_worm_df.mloc(worms,[measure],select_times)
	times=[float(i) for i in select_times]

	slopes=[]
	for i in range(0, len(data)):
		slope,intercept,r,p,std_err=scipy.stats.linregress(times, data[i][0])
		slopes.append(slope)
	slopes=numpy.asarray(slopes)
	lifespans=lifespans[numpy.isnan(slopes)==False]
	slopes=slopes[numpy.isnan(slopes)==False]

	plt.style.use('seaborn-white')
	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	plt.scatter(slopes,lifespans)
	correlation=scipy.stats.pearsonr(slopes, lifespans)
	correlation = numpy.asarray(correlation)
	(m,b) = polyfit(slopes, lifespans, 1)
	yp=polyval([m,b], slopes)
	plot(slopes,yp)
	if correlation[1]<.00001:
		p="p<.00001"
	else:
		p="p=" + ''+ (str)(round(correlation[1],5))
	ftext="r^2="+(str)(round(correlation[0],2)**2)+" "+p
	plt.figtext(.5,.85,ftext,fontsize=12,ha='left')	
	plt.show()

def make_trace(measure, complete_worm_df, high_percentile=0,low_percentile=0,range1=None, range2=None):
	times=complete_worm_df.times
	worms=complete_worm_df.worms
	lifespans=sd.get_lifespans(complete_worm_df)/24
	select_times_long=[]
	select_times_short=[]
	long_lived_worms=[]
	short_lived_worms=[]
	long_lifespans=[]
	short_lifespans=[]

	if range1 ==None:
		for i in range(0, len(lifespans)):
			if lifespans[i]>=numpy.percentile(lifespans,high_percentile):
				long_lived_worms.append(worms[i])
				long_lifespans.append(lifespans[i])
	else:
		for i in range(0, len(lifespans)):
			if lifespans[i]>=numpy.percentile(lifespans,range1[0]) and lifespans[i]<=numpy.percentile(lifespans,range1[1]):
				long_lived_worms.append(worms[i])
				long_lifespans.append(lifespans[i])			
	if range2 ==None:
		for i in range(0, len(lifespans)):			
			if lifespans[i]<=numpy.percentile(lifespans,low_percentile):
				short_lived_worms.append(worms[i])
				short_lifespans.append(lifespans[i])
	else:
		for i in range(0, len(lifespans)):			
			if lifespans[i]>=numpy.percentile(lifespans,range2[0]) and lifespans[i]<=numpy.percentile(lifespans,range2[1]):
				short_lived_worms.append(worms[i])
				short_lifespans.append(lifespans[i])		
	

	for i in range(0, len(times)):
		if (float)(times[i])<(min(long_lifespans)-.1):
				select_times_long.append(times[i])
		if (float)(times[i])<(min(short_lifespans)-.1):
				select_times_short.append(times[i])		 	
	
					
	long_data=complete_worm_df.mloc(long_lived_worms, [measure], select_times_long)
	short_data=complete_worm_df.mloc(short_lived_worms, [measure], select_times_short)	
	
	
	long_average=numpy.sum(long_data, axis=0)/len(long_lived_worms)	
	short_average=numpy.sum(short_data, axis=0)/len(short_lived_worms)
	
	plt.style.use('seaborn-white')
	plt.plot(select_times_long, long_average[0],'darkorchid' ,select_times_short, short_average[0], 'hotpink') 
	if high_percentile != None:
		purple_patch=mpatches.Patch(color='darkorchid', label="Top "+(str)(high_percentile)+"th percentile lifespan cohort")
	if low_percentile !=None:	
		pink_patch=mpatches.Patch(color='hotpink', label="Bottom "+(str)(low_percentile)+"th percentile lifespan cohort")
	if range1 !=None:
		purple_patch=mpatches.Patch(color='darkorchid', label=(str)(range1[0])+" to " + (str)(range1[1]) +"th percentile lifespan cohort")
	if range2 !=None:
		pink_patch=mpatches.Patch(color='hotpink', label=(str)(range2[0])+" to " + (str)(range2[1]) +"th percentile lifespan cohort")		
	plt.legend(handles=[purple_patch, pink_patch], loc='upper right')
	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	plt.ion()
	plt.show()
	plt.gcf().clf
	return long_average, short_average, select_times_short, select_times_long	
def run_stats(x_list, y_list, title, x_label, y_label, out_dir, save_name,control=None):

	if control !=None:
		slope_x, intercept_x,rvalue_x,pvalue_x,stderror_x=scipy.stats.linregress(x=control,y=x_list)
		slope_y,intercept_y,rvalue_y,pvalue_y,stderror_y=scipy.stats.linregress(x=control,y=y_list)
		obsvalues_x=x_list
		predvalues_x=control*slope_x+intercept_x
		residuals_x=predvalues_x-obsvalues_x
		obsvalues_y=y_list
		predvalues_y=control*slope_y+intercept_y
		residuals_y=predvalues_y-obsvalues_y
		correlation=scipy.stats.pearsonr(residuals_x,residuals_y)
		correlation = numpy.asarray(correlation)
		spearman_correlation=scipy.stats.spearmanr(residuals_x, residuals_y)
		spearman_correlation=numpy.asarray(spearman_correlation)
		(m,b) = polyfit(residuals_x, residuals_y, 1)
		yp=polyval([m,b], residuals_x)
		plt.scatter(residuals_x, residuals_y, c='deepskyblue', edgecolors='deepskyblue')
		plot(residuals_x,yp, c='deepskyblue')
		plt.ylim(-4,4)
	
	else:	
		correlation=scipy.stats.pearsonr(x_list, y_list)
		correlation = numpy.asarray(correlation)
		spearman_correlation=scipy.stats.spearmanr(x_list, y_list)
		spearman_correlation=numpy.asarray(spearman_correlation)
		(m,b) = polyfit(x_list, y_list, 1)
		yp=polyval([m,b], x_list)
		plt.scatter(x_list, y_list, c='deepskyblue', edgecolors='deepskyblue')
		plot(x_list,yp, c='deepskyblue')
		#plt.ylim(0,1)
	plt.style.use('seaborn-white')
	
	
	plt.title(title,y=1.05,fontdict={'size':14,'family':'calibri'})
	plt.xlabel(x_label,fontdict={'size':12,'family':'calibri'})
	plt.ylabel(y_label,fontdict={'size':12,'family':'calibri'})
	#plt.ylim((7,20))
	axis=plt.gca()
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
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
	plt.figtext(.5,.85,ftext,fontsize=12,ha='left', family='calibri')
	plt.figtext(.15,.8,gtext,fontsize=12,ha='left',family='calibri')
	#plt.figtext(.7,.88,etext,fontsize=11,ha='left')
	plt.figtext(.8, .2, more_text, fontsize=12, ha='left',family='calibri')
	plt.gcf()
	plt.savefig(out_dir+'/'+save_name+'.png')
	plt.show()
	plt.gcf().clf
	return (correlation[0])