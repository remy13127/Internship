import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from lmfit import Model,Parameter,Parameters
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm, ks_2samp
import random
import mahotas as mh


def MSD_tracks(data,min_length,max_length):
	tracklist = data.TRACK_ID.unique()
	rho_tracks = []
	N_array = []
	for t_id in tracklist:
		
		track_id = data["TRACK_ID"] == t_id
		x = data[track_id]["POSITION_X"].to_numpy()
		y = data[track_id]["POSITION_Y"].to_numpy()

		N = len(x)	
		rho_i = []

		if N>min_length and N<max_length:
			N_array.append(N)
			for n in range(1,N):
				s = 0
				for i in range(0,N-n):
					#r_in = np.sqrt(x[i+n]**2 + y[i+n]**2)
					#r_i  = np.sqrt(x[i]**2   + y[i]**2)
					#s+=(r_in - r_i)**2
					s+=(x[n+i] - x[i])**2 + (y[n+i] - y[i])**2
				rho_i.append(1/(N - n)*s)
			rho_tracks.append(rho_i)


	square_matrix = np.zeros([len(rho_tracks),len(max(rho_tracks,key = lambda x: len(x)))])
	for i,j in enumerate(rho_tracks):
    		square_matrix[i][0:len(j)] = j

	return(square_matrix,N_array)

def MSD_tracks_from_file(filename):
	msd_tracks = []
	with open(filename) as f:
		for line in f:
				inner_list = [float(elt.strip()) for elt in line.split(' ')[:-1]]
				msd_tracks.append(inner_list)
	N_array = [np.shape(msd_tracks[k][:])[0]+1 for k in range(np.shape(msd_tracks)[0])]
	matrix = np.zeros([len(msd_tracks),len(max(msd_tracks,key = lambda x: len(x)))])
	for i,j in enumerate(msd_tracks):
		matrix[i][0:len(j)] = j
	return(matrix,N_array)


def average(data):

	'''This function takes as input the individual MSD curves computed for each track. 
	The output is an averaged MSD curve over all of the available data 
	(the larger the timelag, the smaller the quantity of data available and the poorer the average.'''

	N=np.shape(data)[1]+1
	N_T=np.shape(data)[0]

	VARIANCE = []
	MSD = []

	for k in range(0,N-1):
		s1=0
		s2=0
		N_nonzero=0
		for l in range(0,N_T):
			if data[l][k]!=0.0:
				N_nonzero+=1
			s1+=data[l][k]**2
			s2+=data[l][k]
		VARIANCE.append(1/float(N_nonzero)*s1 - (1/float(N_nonzero)*s2)**2)
		MSD.append(1/float(N_nonzero)*s2)
	return(MSD,VARIANCE)

def COVARIANCE(data,cutoff):
	N = np.shape(data)[1]+1
	covariance_matrix = np.zeros((cutoff,cutoff))
	N_T = np.shape(data)[0]
	for n in range(0,cutoff):
		for m in range(0,cutoff):
			s1=0
			s2=0
			s3=0
			Nnonzeron=0
			Nnonzerom=0
			Nnonzeronm=0
			for i in range(0,N_T):
				if data[i][n]!=0.0 and data[i][m]!=0.0:
					Nnonzeronm+=1
				if data[i][n]!=0.0:
					Nnonzeron+=1
				if data[i][m]!=0.0:
					Nnonzerom+=1
				s1+=data[i][n]*data[i][m]
				s2+=data[i][n]
				s3+=data[i][m]
			covariance_matrix[n][m] = 1/float(Nnonzeronm)*s1 - 1/float(Nnonzeron)*1/float(Nnonzerom)*s2*s3
	return(covariance_matrix)


def NormMSDSlopeError_exp(N,sigma,D,dt,variance_matrix,covariance_matrix,Pmin):
	alpha = 4*D*dt
	x = sigma**2 / (D*dt)
	#Set all sums to 0.0
	sum0=0
	sum1=0
	sum2=0
	sumh0=0
	sumh1=0
	sumh2=0
	if Pmin<2:
		return("Pmin smaller than 2. Abort.")
	elif Pmin>N:
		return("Pmin larger than N. Abort.")
	else:
		for i in range(1,Pmin+1):
			fi = alpha**2 / variance_matrix[i-1]
			sum0+=fi
			sum1+=i*fi
			sum2+=i**2*fi
			h0=0
			h1=0
			for j in range(1,i):
				fj = alpha**2 / variance_matrix[j-1]
				gij = covariance_matrix[i-1,j-1]/alpha**2
				hij = fi*fj*gij
				h0+=hij
				h1+=j*hij
			sumh0+=h0
			sumh1+=i*h0 + h1
			sumh2+=i*h1
		delta = sum0*sum2 - sum1**2
		temp = 1/delta*(sum0 + 2*(sum1**2*sumh0 - sum0*sum1*sumh1 + sum0**2*sumh2)/delta)
		norm_sigmab = np.sqrt(temp)
		return(norm_sigmab)

def NormMSDInterceptError_exp(N,sigma,D,dt,variance_matrix,covariance_matrix,Pmin):
	alpha = 4*D*dt
	x = sigma**2 / (D*dt)
	#Set all sums to 0.0
	sum0=0
	sum1=0
	sum2=0
	sumh0=0
	sumh1=0
	sumh2=0
	if Pmin<2:
		return("Pmin smaller than 2. Abort.")
	elif Pmin>N:
		return("Pmin larger than N. Abort.")
	else:
		for i in range(1,Pmin+1):
			fi = alpha**2 / variance_matrix[i-1]
			sum0+=fi
			sum1+=i*fi
			sum2+=i**2*fi
			h0=0
			h1=0
			for j in range(1,i):
				fj = alpha**2 / variance_matrix[j-1]
				gij = covariance_matrix[i-1,j-1]/alpha**2
				hij = fi*fj*gij
				h0+=hij
				h1+=j*hij
			sumh0+=h0
			sumh1+=i*h0 + h1
			sumh2+=i*h1
		delta = sum0*sum2 - sum1**2
		temp = 1/delta*(sum2 + 2*(sum2**2*sumh0 - sum1*sum2*sumh1 + sum1**2*sumh2)/delta)
		norm_sigmaa = np.sqrt(temp)/x
		return(norm_sigmaa)

##############################################################
##############################################################


def f(n,N,x):
    '''
    This function is an intermediate step in computing the variance of an ideal trajectory. 
    n: time lag
    N: total number of time steps
    x: reduced localization error (x=epsilon/alpha)
    '''
    K = N-n
    if n<=K:
        fminus = n*(4*pow(n,2)*K + 2*K - pow(n,3) + n)/6/pow(K,2) + (2*n*x + (1+(1 - n/K)/2)*pow(x,2))/K
        return(1/fminus)
    else:
        fplus = (6*pow(n,2)*K - 4*n*pow(K,2) + 4*n + pow(K,3) - K)/6/K + (2*n*x + pow(x,2))/K
        return(1/fplus)

def THEORETICAL_VARIANCE(N,sigma,D,dt):
	''' This program computes the theoretical variance for an ideal trajectory. 
	N: total number of time steps
	sigma: localization uncertainty due to noise (identical in X and Y)
	D: diffusion coefficient (measured)
	t: total duration of simulation
	'''
	alpha = 4*D*dt
	x = 4*sigma**2 / alpha
	sigma_array=np.zeros(N-1)
	for n in range(0,N-1):
		sigma_array[n]=alpha**2 / f(n+1,N,x)
	return(sigma_array)

def THEORETICAL_VARIANCE_2(N_array,sigma,D,dt):
	''' This program computes the theoretical variance for an ideal trajectory. 
	N: total number of time steps
	sigma: localization uncertainty due to noise (identical in X and Y)
	D: diffusion coefficient (measured)
	t: total duration of simulation
	'''
	alpha = 4*D*dt
	x = 4*sigma**2 / alpha
	sigma_ensemble=[]
	for k in N_array:
		N=k
		sigma_array = []
		for n in range(0,N-1):
			sigma_array.append(alpha**2 / f(n+1,N,x))
		sigma_ensemble.append(sigma_array)

	square_matrix = np.zeros([len(sigma_ensemble),len(max(sigma_ensemble,key = lambda x: len(x)))])
	for i,j in enumerate(sigma_ensemble):
		square_matrix[i][0:len(j)] = j

	result,useless = average(square_matrix)
	return(result)

def THEORETICAL_COVARIANCE_2(N_array,sigma,D,dt,cutoff):
	"""Cross correlation variance sigma_{nm}^2 
	N: total number of time steps
	sigma: localization uncertainty
	D: Diffusion coefficient
	t: duration of experiment / simulation 
	"""
	epsilon=4*sigma**2
	alpha=4*D*dt
	covariance_ensemble = []
	for k in N_array:
		N=k
		theory_covariance = np.zeros((N-1,N-1))
		for n in range(0,N-1):
			for m in range(0,N-1):
				theory_covariance[n,m] = THEORETICAL_COVARIANCE(n+1,m+1,N,sigma,D,dt)
				theory_covariance[m,n] = theory_covariance[n,m]
		covariance_ensemble.append(theory_covariance)
	
	N_tracks = np.shape(covariance_ensemble)[0]

	#Nsize = np.amax(N_array)-1
	Nsize = cutoff
	temp = np.zeros((N_tracks,Nsize,Nsize)) #(Nloops,N-1,N-1)
	for k in range(0,N_tracks):
		length = np.shape(covariance_ensemble[k][:][:])[0]
		if length<Nsize:
			nshape=length
		else:
			nshape=Nsize
		for n in range(0,nshape):
			for m in range(0,nshape):
				temp[k][n][m] = covariance_ensemble[k][n][m]
	

	result = np.zeros((Nsize,Nsize))
	for n in range(0,Nsize):
		for m in range(0,Nsize):
			Nnonzero = 0
			s=0
			for k in range(0,N_tracks):
				if temp[k][n][m]!=0.0:
					Nnonzero+=1
				s+=temp[k][n][m]
			#if Nnonzero==0.0:
			#	print(n,m,k)
			result[n][m]=s/float(Nnonzero)
	return(result)


def THEORETICAL_COVARIANCE(m,n,N,sigma,D,dt):
	"""Cross correlation variance sigma_{nm}^2 
	N: total number of time steps
	sigma: localization uncertainty
	D: Diffusion coefficient
	t: duration of experiment / simulation 
	"""
	epsilon=4*sigma**2
	alpha=4*D*dt
	K = N - n
	P = N - m
	if m<=n:
		temp = m
		m = n
		n = temp
	if m+n<=N:
		sigmanm=n/(6*K*P)*(4*pow(n,2)*K + 2*K - pow(n,3) + n + (m-n)*(6*n*P - 4*pow(n,2) - 2))*pow(alpha,2)+1/K*(2*n*alpha*epsilon+(1-n/(2*P))*pow(epsilon,2)/2)
		return(sigmanm)
	else:
		sigmanm=1/(6*K)*(6*pow(n,2)*K - 4*n*pow(K,2)+pow(K,3)+4*n - K +(m-n)*((n+m)*(2*K+P)+2*n*P-3*pow(K,2)+1))*pow(alpha,2)+1/K*(2*n*alpha*epsilon+pow(epsilon,2)/2)
		return(sigmanm)


def g(m,n,N,sigma,D,dt):
	alpha = 4*D*dt
	return(THEORETICAL_COVARIANCE(m,n,N,sigma,D,dt) / alpha**2)

def NormMSDSlopeError(N,sigma,D,dt,Pmin):
	x = sigma**2 / (D*dt)
	#Set all sums to 0.0
	sum0=0
	sum1=0
	sum2=0
	sumh0=0
	sumh1=0
	sumh2=0
	if Pmin<2:
		return("Pmin smaller than 2. Abort.")
	elif Pmin>N:
		return("Pmin larger than N. Abort.")
	else:
		for i in range(1,Pmin+1):
			fi = f(i,N,x)
			sum0+=fi
			sum1+=i*fi
			sum2+=i**2*fi
			h0=0
			h1=0
			for j in range(1,i):
				fj = f(j,N,x)
				gij = g(i,j,N,sigma,D,dt)
				hij = fi*fj*gij
				h0+=hij
				h1+=j*hij
			sumh0+=h0
			sumh1+=i*h0 + h1
			sumh2+=i*h1
		delta = sum0*sum2 - sum1**2
		temp = 1/delta*(sum0 + 2*(sum1**2*sumh0 - sum0*sum1*sumh1 + sum0**2*sumh2)/delta)
		#print("delta = ",delta,"temp = ",temp)
		norm_sigmab = np.sqrt(temp)
		return(norm_sigmab)

def NormMSDInterceptError(N,sigma,D,dt,Pmin):
	x = sigma**2 / (D*dt)
	#Set all sums to 0.0
	sum0=0
	sum1=0
	sum2=0
	sumh0=0
	sumh1=0
	sumh2=0
	if Pmin<2:
		return("Pmin smaller than 2. Abort.")
	elif Pmin>N:
		return("Pmin larger than N. Abort.")
	else:
		for i in range(1,Pmin+1):
			fi = f(i,N,x)
			sum0+=fi
			sum1+=i*fi
			sum2+=i**2*fi
			h0=0
			h1=0
			for j in range(1,i):
				fj = f(j,N,x)
				gij = g(i,j,N,sigma,D,dt)
				hij = fi*fj*gij
				h0+=hij
				h1+=j*hij
			sumh0+=h0
			sumh1+=i*h0 + h1
			sumh2+=i*h1
		delta = sum0*sum2 - sum1**2
		temp = 1/delta*(sum2 + 2*(sum2**2*sumh0 - sum1*sum2*sumh1 + sum1**2*sumh2)/delta)
		#print("delta = ",delta,"temp = ",temp)
		norm_sigmaa = np.sqrt(temp)/x
		return(norm_sigmaa)

#########################################################################
#########################################################################

def BROWNIAN_MOTION(N,dimension,D,t):
	'''This simulation generates 2D isotropic Brownian motion trajectories'''
	dt = t / float(N)
	x = np.zeros ([dimension, N])
	for j in range (1, N):
		s = np.sqrt(2.0*dimension*D*dt)*np.random.randn(1)
		if (dimension==1):
			dx=s*np.ones(1)
		else:
			dx = np.random.randn(dimension)
			norm_dx = np.sqrt(np.sum(dx ** 2))
			for i in range(0, dimension):
				dx[i] = s * dx[i] / norm_dx
		x[0:dimension,j] = x[0:dimension,j-1] + dx[0:dimension]
	return(x)


def msd_average(x,y):
	x=np.array(x)
	y=np.array(y)
	N = len(x)
	rhoarray = []
	for n in range(1,N):
		K = N - n
		rhon=0
		for i in range(0,K):
			rhon+=(x[n+i] - x[i])**2 + (y[n+i] - y[i])**2
		rhoarray.append(1/K*rhon)
	return(rhoarray)


#########################
#########################

def data_histogram(data,dt):
	tracklist = data.TRACK_ID.unique()  
	x = []
	length_track = []
	index = 0
	for tid in tracklist:
		trackid = data["TRACK_ID"] == tid
		x.append(data[trackid]["POSITION_X"].to_numpy())
		length_track.append(np.shape(x[index][:])[0])
		index+=1
	return(length_track)

def plot_preprocessed_dist(processed_feat,feat):
	fig = plt.figure(figsize=[18.,5.])
	fig.tight_layout()
	gs = GridSpec(1,4)

	dist_alpha = fig.add_subplot(gs[0])
	dist_D = fig.add_subplot(gs[1])
	dist_confinement  = fig.add_subplot(gs[2])
	#intensity_length = fig.add_subplot(gs[3])
	cdf_d = fig.add_subplot(gs[3])

	dist_alpha.hist(processed_feat[:,0],bins=10)
	dist_alpha.hist(feat[:,0],bins=10,color='r', ec='r',alpha=0.5)
	dist_alpha.set_xlabel(r'$\alpha$')
	dist_alpha.set_ylabel('#')
	dist_alpha.spines["top"].set_visible(False)  
	dist_alpha.spines["right"].set_visible(False)
	dist_alpha.get_xaxis().tick_bottom()  
	dist_alpha.get_yaxis().tick_left()
	dist_alpha.xaxis.set_minor_locator(MultipleLocator(1))
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)

	dist_D.hist(processed_feat[:,1],bins=10)
	dist_D.hist(feat[:,1],bins=10,color='r', ec='r',alpha=0.5)
	dist_D.set_xlabel(r'$D$')
	dist_D.set_ylabel('#')
	dist_D.spines["top"].set_visible(False)  
	dist_D.spines["right"].set_visible(False)
	dist_D.get_xaxis().tick_bottom()  
	dist_D.get_yaxis().tick_left()
	dist_D.xaxis.set_minor_locator(MultipleLocator(1))
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)

	dist_confinement.hist(processed_feat[:,2],bins=10)
	dist_confinement.hist(feat[:,2],bins=10,color='r', ec='r',alpha=0.5)
	dist_confinement.set_xlabel('Confinement ratio')
	dist_confinement.set_ylabel('#')
	dist_confinement.spines["top"].set_visible(False)  
	dist_confinement.spines["right"].set_visible(False)
	dist_confinement.get_xaxis().tick_bottom()  
	dist_confinement.get_yaxis().tick_left()
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)


	#intensity_length.scatter(feat[:,3],feat[:,2])
	#intensity_length.set_xlabel("Track length")
	#intensity_length.set_ylabel('Mean intensity variation / frame (%)')
	#intensity_length.spines["top"].set_visible(False)  
	#intensity_length.spines["right"].set_visible(False)

	darray = np.sort(np.array(feat[:,1]))
	Darray = darray[darray > 1.0E-07]
	n, bins, patches = cdf_d.hist(Darray, 10000, density=True, histtype='step',cumulative=True, label='Empirical')
	cdf_d.set_xlabel(r'$D$')
	cdf_d.set_ylabel('Cumulative distribution function')
	cdf_d.spines["top"].set_visible(False)  
	cdf_d.spines["right"].set_visible(False)
	cdf_d.get_xaxis().tick_bottom()  
	cdf_d.get_yaxis().tick_left()
	cdf_d.set_xscale('log')
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)

	plt.subplots_adjust(wspace=0.8)
	plt.show()

def plot_track_characteristics(dt,feat,msd_all,nbr_frames,color_array,minframe,maxframe,TRACKS):

	alpha,diff,conf,length_track = zip(*feat)

	fig = plt.figure(figsize=[18.,8.])
	fig.tight_layout()
	gs = GridSpec(4,7)

	ax_joint = fig.add_subplot(gs[1:4,3:6])
	ax_marg_x = fig.add_subplot(gs[0,3:6])
	ax_marg_y = fig.add_subplot(gs[1:4,6])
	msd_plot  = fig.add_subplot(gs[1:4,0:3])
	hist_frame = fig.add_subplot(gs[0,0:3])
	trajectories = fig.add_subplot(gs[0,6])

	ax_joint.scatter(alpha,diff,c=color_array)
	#ax_joint.set_yscale('log')
	#ax_joint.text(max(alpha)/50,max(diff)/2,"Mean D = "+str(np.mean(diff)))
	#ax_joint.text(max(alpha)/50,max(diff)/2.3,r"Mean $\alpha$ = "+str(np.mean(alpha)))

	ax_marg_x.axis('on')
	ax_marg_x.spines["top"].set_visible(False)  
	ax_marg_x.spines["right"].set_visible(False)
	ax_marg_x.hist(alpha,bins=70,orientation="vertical",color='grey', ec='grey')
	ax_marg_y.axis('on')
	ax_marg_y.spines["top"].set_visible(False)  
	ax_marg_y.spines["right"].set_visible(False)
	ax_marg_y.hist(diff,orientation="horizontal",bins=np.logspace(np.log10(min(diff)),np.log10(max(diff)), 30),color='grey', ec='grey')
	ax_marg_y.set_yscale('log')

	# Turn off tick labels on marginals
	plt.setp(ax_marg_x.get_xticklabels(), visible=True)
	plt.setp(ax_marg_y.get_yticklabels(), visible=True)

	# Set labels on joint
	ax_joint.set_ylabel(r'$D$')
	ax_joint.set_xlabel(r'$\alpha$')
	ax_joint.spines["top"].set_visible(False)  
	ax_joint.spines["right"].set_visible(False)
	ax_joint.get_xaxis().tick_bottom()  
	ax_joint.get_yaxis().tick_left()
	ax_joint.set_yscale('log')
	ax_joint.set_ylim(min(feat[:,1]),max(feat[:,1]))
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)


	#MSD
	N_T = np.shape(msd_all)[0]
	for k in range(N_T):
	    N=len(msd_all[k][:])
	    t = [n*dt for n in np.linspace(1,N,N)]
	    msd_plot.plot(t,msd_all[k][:],c=color_array[k])
	    
	    
	msd_plot.set_ylabel(r"Mean square displacement ($\mu$m$^2/$s)")    
	msd_plot.set_xlabel("Time lag (s)")
	msd_plot.spines["top"].set_visible(False)  
	msd_plot.spines["right"].set_visible(False)
	msd_plot.get_xaxis().tick_bottom()  
	msd_plot.get_yaxis().tick_left()
	msd_plot.set_yscale('log')
	msd_plot.set_xscale('log')
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)


	#Hist frames

	hist_frame.hist(nbr_frames,bins=100,range=(0, 100))
	hist_frame.axvline(x=maxframe, ymin=0, ymax=1,linestyle="--",color="red")
	hist_frame.axvline(x=minframe, ymin=0, ymax=1,linestyle="--",color="red")
	hist_frame.set_xlabel('N frames')
	hist_frame.set_ylabel('#')
	hist_frame.text(50,300,'Minimum number of frames = '+str(minframe))
	hist_frame.text(50,250,'Maximum number of frames = '+str(maxframe))
	hist_frame.spines["top"].set_visible(False)  
	hist_frame.spines["right"].set_visible(False)
	hist_frame.get_xaxis().tick_bottom()  
	hist_frame.get_yaxis().tick_left()
	#hist_frame.xaxis.set_minor_locator(MultipleLocator(1))
	plt.xticks(fontsize=10)  
	plt.yticks(fontsize=10)

	clustercolor = np.unique(color_array)
	for color in clustercolor:
		for k in range(len(color_array)):
			if color_array[k]==color:
				trajectories.plot(TRACKS[k][0],TRACKS[k][1],c=color)
	trajectories.set_xticks([])
	trajectories.set_yticks([])
	
	plt.subplots_adjust(wspace=0.8)
	#plt.savefig(output_folder+"/data_characteristics.png")
	plt.show()

def mmsd_plot(timelag,mmsd,mvar):
	plt.figure(figsize=(8, 6))
	ax = plt.subplot(111)  
	ax.spines["top"].set_visible(False)  
	ax.spines["right"].set_visible(False)
	ax.xaxis.grid(True,which='both')
	ax.get_xaxis().tick_bottom()  
	ax.get_yaxis().tick_left()
	plt.xticks(fontsize=14)  
	plt.yticks(fontsize=14)
	plt.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(mmsd,mvar)],[a + np.sqrt(b) for a,b in zip(mmsd,mvar)], color="#3F5D7D") 
	plt.plot(timelag,mmsd,color="white", lw=2)
	plt.ylabel(r"Mean square displacement ($\mu$m$^2/$s)",fontsize=16)
	plt.xlabel('Time Lag (n)',fontsize=16)
	plt.show()

def cutoff_function(dt,N,data,timelag,mmsd,mvar):
	cutoff = int(input("After how many time steps n do you want to cut the data? ")) # crop data 
	if cutoff > N-1:
		print("Your cutoff is larger than the available data !")
	else:
		print("You have set the cutoff to a time lag of ",str(cutoff)," steps...")

	data = data[:,:cutoff]
	timelag = timelag[:cutoff]
	mmsd = mmsd[:cutoff]
	mvar = mvar[:cutoff]
	return(data,timelag,mmsd,mvar,cutoff)

def msd(t, D, alpha):
	"""
	Parameters:
	D<float>: diffusion coefficient
	alpha<float>: power exponent
	
	Returns:
	4*D*t^alpha
	"""
	return(4*D*t**alpha)

def data_pool(files,dt,minframe=5,maxframe=500,rsquared_threshold=0.8):
	"""
	This function extracts trajectories from CSV files, computes the MSD curves. The MSD curves are fitted according to a 4*D*t^alpha power law, where D is the diffusion coefficient. The fit parameters are stored in the DATA list if R² > rsquared_threshold. DATA contains for each surviving track the power exponent alpha, the diffusion coefficient D, the confinement ratio c, the tracklength, the track ID, the x sequence of coordinates, the y sequence of coordinates. 
	
	Parameters:
	files<list of str>: list of csv filenames which will be analyzed.
	dt<float>: time interval bewteen two frames.
	minframe(default=5),maxframe(default=500)<int>: minimum (and maximum) number of frames in order to keep a track and perform the analysis. 
	rsquared_threshold(default=0.8)<float>: threshold on the accurateness of the fit. 
	
	Returns:
	DATA<list>: contains [alpha<float>,D<float>,confinement ratio<float>,number of frames<int>,track ID,x<ndarray>,y<ndarray>] for each retained trajectory. 
	"""

	minalpha = 1.0E-03
	minD = 1.0E-04
	maxD = 4
	maxalpha = 3

	msd_model = Model(msd)
	params = Parameters()
	params['alpha']   = Parameter(name='alpha', value=1.0, min=minalpha,max=maxalpha)
	params['D']   = Parameter(name='D', value=0.1, min=minD,max=maxD)

	DATA = []
	for filename in files:

		data = pd.read_csv(filename) 
		tracklist = data.TRACK_ID.unique()  #list of track ID in data

		for tid in tracklist:

			trackid = data["TRACK_ID"] == tid
			x = data[trackid]["POSITION_X"].to_numpy()   #x associated to track 'tid'
			y = data[trackid]["POSITION_Y"].to_numpy()   #y associated to track 'tid'

			rhon = []
			if len(x)<maxframe and len(x)>minframe:
				for n in range(1,len(x)):             #for each n = each time lag
					s = 0
					for i in range(0,len(x)-n):
						s+=(x[n+i] - x[i])**2 + (y[n+i] - y[i])**2
					rhon.append(1/(len(x)-n)*s)

				N = len(rhon)+1
				t = [n*dt for n in np.linspace(1,N-1,N-1)]

				nbrpts = int(0.3*N)
				result = msd_model.fit(rhon[:nbrpts+1], params, t=t[:nbrpts+1])
			    
				s=0
				for p in range(0,len(x)-1):
					s+= np.sqrt((x[p+1]-x[p])**2+(y[p+1]-y[p])**2)
				confinement_ratio = np.sqrt((x[-1]-x[0])**2+(y[-1]-y[0])**2)/s

				alpha = result.params['alpha'].value
				D = result.params['D'].value
				rsquare = 1 - result.residual.var() / np.var(rhon[:nbrpts+1])

				if rsquare > rsquared_threshold and confinement_ratio!=0.0:
					feat = [alpha,D,confinement_ratio,len(x),tid,x,y]
					DATA.append(feat)
	return(DATA)

def two_distributions_plot(DATA1,DATA2,label1,label2):
	fig = plt.figure(figsize=(16, 4))
	grid = plt.GridSpec(2, 5, hspace=0.4, wspace=0.5)
	Dvsalpha = fig.add_subplot(grid[:,0:2])
	distD = fig.add_subplot(grid[0,2])
	logdistD = fig.add_subplot(grid[0,3])
	cdfD = fig.add_subplot(grid[0,4])
	distA = fig.add_subplot(grid[1,2])
	logdistA = fig.add_subplot(grid[1,3])
	cdfA = fig.add_subplot(grid[1,4])
	
	D1 = []
	A1 = []
	for k in range(np.shape(DATA1)[0]):
		D1.append(DATA1[k][1])
		A1.append(DATA1[k][0])
    
	D2 = []
	A2 = []
	for k in range(np.shape(DATA2)[0]):
		D2.append(DATA2[k][1])
		A2.append(DATA2[k][0])
		
	Dvsalpha.scatter(A1,D1,label=label1,alpha=0.5)
	Dvsalpha.scatter(A2,D2,label=label2,alpha=0.5)
	Dvsalpha.set_ylim(min(D1+D2),max(D1+D2))
	Dvsalpha.set_yscale('log')
	Dvsalpha.set_xlabel(r'$\alpha$')
	Dvsalpha.set_ylabel('D')
	Dvsalpha.spines["top"].set_visible(False)  
	Dvsalpha.spines["right"].set_visible(False)
	Dvsalpha.legend()
	   
	distD.hist(D1,bins=30,alpha=0.5,label=label1)
	distD.hist(D2,bins=30,alpha=0.5,label=label2)
	distD.set_xlabel('D')
	distD.set_ylabel('#')
	#distD.legend(fontsize='small')
	distD.spines["top"].set_visible(False)  
	distD.spines["right"].set_visible(False)
	
	distA.hist(A1,bins=30,alpha=0.5,label=label1)
	distA.hist(A2,bins=30,alpha=0.5,label=label2)
	distA.set_xlabel(r'$\alpha$')
	distA.set_ylabel('#')
	#distA.legend(fontsize='small')
	distA.spines["top"].set_visible(False)  
	distA.spines["right"].set_visible(False)	

	cdfD.hist(D1,1000, density=True, cumulative=True, histtype='step', alpha=0.8,label=label1)
	cdfD.hist(D2,1000, density=True, cumulative=True, histtype='step', alpha=0.8,label=label2)
	cdfD.set_xscale('log')
	cdfD.set_xlabel('D')
	cdfD.set_ylabel('CDF')
	#cdfD.legend(fontsize='small')
	cdfD.spines["top"].set_visible(False)  
	cdfD.spines["right"].set_visible(False)
	
	cdfA.hist(A1,1000, density=True, cumulative=True, histtype='step', alpha=0.8,label=label1)
	cdfA.hist(A2,1000, density=True, cumulative=True, histtype='step', alpha=0.8,label=label2)
	cdfA.set_xscale('log')
	cdfA.set_xlabel(r'$\alpha$')
	cdfA.set_ylabel('CDF')
	#cdfA.legend(fontsize='small')
	cdfA.spines["top"].set_visible(False)  
	cdfA.spines["right"].set_visible(False)

	def logbin(x, bins):
		hist, bins = np.histogram(x, bins=bins)
		logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
		return(logbins)
	
	
	logdistD.hist(D1, bins=logbin(D1,30),label=label1,alpha=0.5)
	logdistD.hist(D2, bins=logbin(D2,30),label=label2,alpha=0.5)
	logdistD.set_xscale('log')
	logdistD.set_xlabel('D')
	logdistD.set_ylabel('#')
	#logdistD.legend(fontsize='small')
	logdistD.spines["top"].set_visible(False)  
	logdistD.spines["right"].set_visible(False)
	
	logdistA.hist(A1, bins=logbin(A1,30),label=label1,alpha=0.5)
	logdistA.hist(A2, bins=logbin(A2,30),label=label2,alpha=0.5)
	logdistA.set_xscale('log')
	logdistA.set_xlabel(r'$\alpha$')
	logdistA.set_ylabel('#')
	#logdistA.legend(fontsize='small')
	logdistA.spines["top"].set_visible(False)  
	logdistA.spines["right"].set_visible(False)
	
	#plt.tight_layout()
	plt.show()

def compute_mean_intensity_map(filename,gauss=5,plot=False):
	"""
	This function takes an image, applies a gaussian filter. Then it labels individual objects detected after a segmentation step. For each object thus labeled, the function averages over the area the original pixel intensity.
	
	Parameters:
	filename<str>: path of the image file.
	gauss<float>(default=5): standard deviation for the gaussian filter.
	plot<bool>(delfaut=False): option to show the segmented mean intensity image.
	
	Returns:
	mean_intensity_map<ndarray>: matrix of the mean pixel intensity per segmented object
	"""
	image = mh.imread(filename)
	imagef = mh.gaussian_filter(image,gauss)
	imagef = imagef.astype('uint8')
	T = mh.thresholding.otsu(imagef)
	labeled,nr_objects = mh.label(imagef > T)
    
	cell_index = np.unique(labeled)
	mean_intensity_map = np.zeros_like(image)

	for lab in cell_index:
		s=0
		n=0
		for i in range(np.shape(image)[0]):
			for j in range(np.shape(image)[1]):
				if labeled[i,j]==lab:
					s+=image[i,j]
					n+=1
		meanI = s/n
		for i in range(np.shape(image)[0]):
			for j in range(np.shape(image)[1]):
				if labeled[i,j]==lab:
					mean_intensity_map[i,j]=meanI
	if plot==True:
		plt.imshow(mean_intensity_map)
		plt.colorbar()
		plt.title('Mean pixel intensity over each cell')
		plt.show()
	return(mean_intensity_map)


def partition(list_in, n):
	"""
	This function randomly shuffles a list and makes n partitions.
	
	Parameters:
	<list or ndarray>list_in: list
	n<int>: number of partitions 
	
	Returns:
	<list> of shape (n,len(list_in/n)) -> element [0] is the first partition and so on.
	"""
	random.shuffle(list_in)
	return([list_in[i::n] for i in range(n)])

def ecdf(data):
	"""
	This function computes the empirical cumulative distribution function for a set of data. 
	Parameters:
	data<ndarray>: list of values
	Returns:
	x_values<ndarray>: bins coordinates for the ECDF
	y_values<ndarray>: ECDF values
	"""
	# create a sorted series of unique data
	cdfx = np.sort(np.unique(data))
	# x-data for the ECDF: evenly spaced sequence of the uniques
	x_values = np.linspace(start=min(cdfx),stop=max(cdfx),num=len(cdfx))
	# size of the x_values
	size_data = data.size
	# y-data for the ECDF:
	y_values = []
	for i in x_values:
		# all the values in raw data less than the ith value in x_values
		temp = data[data <= i]
		# fraction of that value with respect to the size of the x_values
		value = temp.size / size_data
		# pushing the value in the y_values
		y_values.append(value)
	# return both x and y values    
	return(x_values,y_values)



def kolmogorov_smirnov(dist1,dist2,nloop=1000,plot=False):
	"""
	This function computes the KS statistic D and associated p-value between two samples. D is defined as the maximum distance separating the ECDF of each sample. The inital assumption is that both samples were drawn from the same distribution. A p-value under 1 % suggests that the original samples were probably drawn from different distributions.

	Parameters:
	dist1,dist2<list or ndarray>: lists of values.
	nloop<int>(default=1000): number of iterations to compute the KS statistic distribution. 
	plot<bool>(default=False): plot the KS statistic distribution and initial D.
	Returns: 
	stat0,pvalue<float>: KS statistic, p-value.
	
	"""
	stat0,pvalue0 = ks_2samp(dist1,dist2)
	print("KS statistic = ",stat0)
	size1,size2 = len(dist1),len(dist2)
	conc = dist1+dist2
	s=0
	stat_array = []
	for i in range(nloop):
		shuffled1 = partition(conc,1)[0]
		shuffled2 = partition(conc,1)[0]
		dist1 = shuffled1[0:size1]
		dist2 = shuffled2[0:size2]
		stat,pval = ks_2samp(dist1,dist2)
		stat_array.append(stat)
		if stat>=stat0:
			s+=1
	pvalue = s/nloop
	print("bootstrap p-value = ",pvalue)
	if plot==True:
		hist,bin_edge = np.histogram(stat_array,bins=10)
		plt.hist(stat_array,bins=10)
		plt.vlines(stat0,0,max(hist),color='r')
		plt.xlabel(r'KS statistic $D^*$')
		plt.title('Statistical distribution for the K-S statistic')
		plt.show()
	return(stat0,pvalue)
