import numpy
from pylab import *

#timestamped_data is a pandas DataFrame
def make_survival_curves(timestamped_data,x_label,y_label,title,out_dir):

	deaths=[]
	births=[]

	for i in range(0,len(timestamped_data)):
		for item in timestamped_data[i]['Death'].tolist():
			deaths.append(item)
		for item in timestamped_data[i]['Hatch'].tolist():
			births.append(item)
	deaths=numpy.asarray(deaths)
	births=numpy.asarray(births)		
	#calculate lifespans and convert to days
	lifespans=(deaths-births)/(60*60*24)
	total_worms=len(lifespans)

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
