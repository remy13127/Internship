import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from lmfit import Model,Parameter,Parameters
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm, ks_2samp
import random
from sklearn.metrics import r2_score


#########################################################
######### FUNCTIONS FOR NON SPECIFIC ANALYSIS ###########
#########################################################

def T_MSD(x,y,dt=0.05):
	"""Takes x and y trajectories and computes the time-averaged MSD (T-MSD)
	
	Parameters
	----------
	x,y : list or ndarray
	    series of positions separated by a time dt
	dt : float
	    time step between each position in the trajectories
	    
	Returns
	-------
	msd
	   The T-MSD computed at each timelag n*dt, n=1,...,N-1
	timelag
	    The associated time lag array
	"""
	
	msd = []
	N = len(x)
	for n in range(1,N):
		s = 0
		for i in range(0,N-n):
			s+=(x[n+i] - x[i])**2 + (y[n+i] - y[i])**2
		msd.append(1/(N-n)*s)

	timelag = np.linspace(dt,(N-1)*dt,N-1)
	return(msd,timelag)


def pool_data(files,dt,minframe=5,maxframe=10000,rsquared_threshold=0.0,fit_option=2,display_id=False,dataframe=True):
	"""This function reads the trajectories in all of the files, computes the T-MSD, performs a fit to evaluate D and alpha.
	
	Parameters
	----------
	files : list of str
	    The list of the TrackMate CSV filenames, containing columns POSITION_X, POSITION_Y and TRACK_ID
	dt : float
	    time step between each position in the trajectories
	minframe, maxframe : int 
	    filter on the minimum / maximum number of frames per retained trajectory. Default set to respectively 5 and 10000.
	rsquared_threshold : float <= 1
	    filter on the minimum value for the coefficient of determination during the fit of the T-MSDs. Default is 0.0.
	fit_option : int or list or str
	    options for the points to consider during the fit of the T-MSD. If int, the fit is performed over the N first points. If list the fit is performed from the first element to the second. str option available: "thirty_percent". Default is 2.
	display_id : bool, optional
	    prints the ID of the tracks as they are being processed. Useful if the analysis is too slow. Default is disabled.
	dataframe : bool, optional
	    returns a DataFrame instead of a numpy array. The diffusion coefficient is directly transformed to its log10 form. Default is enabled.
	    
	Returns
	-------
	df
	    a DataFrame collecting all of the generated data
	DATA
	    a numpy matrix containing the same information
	"""
	
	#Initial tests:
	if isinstance(files,list)==False:
		print("Please provide a list of TrackMate CSV files, with columns POSITION_X, POSITION_Y and TRACK_ID.")
		return
	else:
		print("Parameters for the MSD analysis: dt = ",dt) 
		print('Initial filters: minframe = ',minframe,', maxframe = ',maxframe,', R2 threshold = ',rsquared_threshold) 
		print("Fit option: ",fit_option)

	#Parameters for the fit
	minalpha = 0.1
	minD = 1.0E-06
	maxD = 1
	maxalpha = 3
	
	def msdlog(logt, D, alpha):
		return(alpha*logt + np.log(4*D))

	
	msdlog_model = Model(msdlog)
	params = Parameters()
	params['alpha']   = Parameter(name='alpha', value=1.0, min=minalpha,max=maxalpha)
	params['D']   = Parameter(name='D', value=0.1, min=minD,max=maxD)
	
	#Interpret fit option that does not depend on the tracklength
	if isinstance(fit_option, int):
		nbrpts = fit_option
		n0 = 0
		
	elif isinstance(fit_option, list):
		nbrpts = fit_option[1]
		n0 = fit_option[0]
	
	DATA = []	#Initialize the DATA output matrix
	print("Reading filenames in ",str(files),'...')
	
	for p in range(len(files)):
		filename = files[p]
		print('Analysis for',filename,'...')
		
		data = pd.read_csv(filename) 
		tracklist = data.TRACK_ID.unique()  #list of track IDs in the file
		conserved_tracks=0
		idxt=0
		for tid in tracklist:

			if display_id==True:
				print('Track '+str(idxt)+' out of '+str(len(tracklist)))
			
			trackid = data["TRACK_ID"] == tid
			x = data[trackid]["POSITION_X"].to_numpy()   #x associated to track 'tid'
			y = data[trackid]["POSITION_Y"].to_numpy()   #y associated to track 'tid'
			spotIDs = data[trackid]["ID"].to_numpy()
			
			N = len(x)
			if N<=maxframe and N>=minframe:
				tmsd,timelag = T_MSD(x,y,dt)
				
				#Interpret fit option
				if fit_option=="thirty_percent":
					nbrpts = round(0.31*N)
					n0 = 0
				
				result = msdlog_model.fit([np.log(k) for k in tmsd[n0:nbrpts]], params, logt=[np.log(k) for k in timelag[n0:nbrpts]])
				alpha = result.params['alpha'].value
				D = result.params['D'].value
				
				N_for_R2 = round(0.6*len(tmsd)) #evaluate R2 on a third of the MSD length
				y_true = tmsd[:N_for_R2]
				y_pred = [4*D*ndt**alpha for ndt in timelag[:N_for_R2]]
				rsquare = r2_score(y_true,y_pred)
				
				c = confinement_ratio(x,y)
				
				if rsquare > rsquared_threshold:
					feat = [alpha,D,c,rsquare,N,x,y,tmsd,filename,spotIDs]
					DATA.append(feat)
					conserved_tracks+=1
			idxt+=1
		print(conserved_tracks," tracks were kept out of ",len(trackid),'. Done.')
	
	if dataframe==True:
		print("Generating a DataFrame...")
		df = pd.DataFrame(DATA,columns=['alpha', 'D', 'c','R2','N','x','y','MSD','Filename','spotIDs'])
		df['D'] = df['D'].map(lambda x: np.log10(x))
		print("End of the program. Returning DataFrame.")
		return(df)
	else:
		print("End of the program. Returning numpy array.")
		return(DATA)
	
def confinement_ratio(x,y):
	s=0
	for p in range(0,len(x)-1):
		s+= np.sqrt((x[p+1]-x[p])**2+(y[p+1]-y[p])**2)
	if s!=0.0:
		return(np.sqrt((x[-1]-x[0])**2+(y[-1]-y[0])**2)/s)
	else:
		return(0.0)

#########################################################
######### FUNCTIONS FOR THE SPECIFIC ANALYSIS ###########
#########################################################

def TE_MSD(MSD_Series,dt=0.05,cutoff=0.05,plot=True,mintracks=2,display_ntracks=False):
	"""Takes a series of T-MSD tracks and computes an ensemble average / variance, the TE-MSD. 
	
	Parameters
	----------
	MSD_Series : list of lists/arrays
	    series of T-MSDs associated to each tracks
	dt : float
	    time step between each T-MSD points
	cutoff : float
	    time lag at which the TE-MSD should be cut (because there not enough tracks to have a good average)
	plot : bool
	    plots the complete TE-MSD, the associated variance, and shows where the cutoff was set
	mintracks : int
	    minimum number of tracks over which the ensemble average is performed to retain a point (CLT -> 30)
	display_ntracks : bool
	    shows how many tracks are available at each time lag. Helps to select either a cutoff or a mintracks.
	    
	Returns
	-------
	TE-MSD
	   The ensemble average of the T-MSDs
	TE-VAR
	    The associated ensemble variance
	timelag
	    The associated time lag'
	"""
	
	# Turn the series of T-MSDs into a square matrix
	matrix = np.zeros([len(MSD_Series),len(max(MSD_Series,key = lambda x: len(x)))])
	for i,j in enumerate(MSD_Series):
		matrix[i][0:len(j)] = j
	
	#Initialize the ensemble average and variance
	TE_MSD = []
	TE_VAR = []

	N_tracks = np.shape(matrix)[0]
	maxlength = np.shape(matrix)[1]
	
	print("Compute the TE-MSD and ensemble variance...")
	#Sum of all of the matrix elements column-wise.
	for k in range(0,maxlength):
		s1=0
		s2=0
		N_nonzero=0
		for l in range(0,N_tracks):
			if matrix[l][k]!=0.0:
				N_nonzero+=1
			s1+=matrix[l][k]**2
			s2+=matrix[l][k]
		if display_ntracks==True:
			print("Number of T-MSDs at time lag ",str(round((k+1)*dt,2))," = ",N_nonzero)
		if N_nonzero>=mintracks:
			TE_VAR.append(N_nonzero/(N_nonzero-1)*(1/float(N_nonzero)*s1 - (1/float(N_nonzero)*s2)**2))
			TE_MSD.append(1/float(N_nonzero)*s2)	

	timelag = np.linspace(1*dt,len(TE_MSD)*dt,len(TE_MSD))
	
	#Check that there are enough tracks to perform a satisfying ensemble average
	if len(TE_MSD)==0:
		print('The number of T-MSD curves is insufficient at all time lags to properly estimate the mean and sample variance. Try to reduce mintracks. The higher the mintracks, the more the central limit theorem is satisfied.')
	
	#Plot the complete TE-MSD with the cutoff position
	if plot==True:	
		plt.plot(timelag,TE_MSD,color='k')
		plt.fill_between(timelag,[msd+np.sqrt(var) for msd,var in zip(TE_MSD,TE_VAR)], [msd-np.sqrt(var) for msd,var in zip(TE_MSD,TE_VAR)],color='gray',alpha=0.1)
		plt.vlines(cutoff,plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyles='dashed',colors='k')
		plt.xlabel('Time lag (s)')
		plt.ylabel(r'TE-MSD ($\mu$m$^2$)')
		plt.show()
	
	# if the user has set a cutoff value
	if cutoff>0.05:
		cut = int(cutoff/dt)
		TE_MSD,TE_VAR,timelag = TE_MSD[:cut],TE_VAR[:cut],timelag[:cut]
		print("You have set a cutoff at ",cutoff," s...")
	print("Done. The TE-MSD and associated variance have been generated.")
	return(TE_MSD,TE_VAR,timelag)

def COVARIANCE(MSD_Series,dt=0.05,cutoff=0.0,plot=False):
	"""Takes a series of T-MSD tracks and computes an ensemble covariance. 
	
	Parameters
	----------
	MSD_Series : list of lists/arrays
	    series of T-MSDs associated to each tracks
	dt : float
	    time step between each T-MSD points. Default is dt=0.05 s.
	cutoff : float
	    time lag at which the covariance matrix should be cut (because there not enough tracks to have a good average, or the computing time might be too long.)
	plot : bool
	    plots the covariance map
	
	Returns
	-------
	covariance_matrix
	   The ensemble covariance of the T-MSDs
	timelag
	    The associated time lag array
	"""


	# Turn the series of T-MSDs into a square matrix
	matrix = np.zeros([len(MSD_Series),len(max(MSD_Series,key = lambda x: len(x)))])
	for i,j in enumerate(MSD_Series):
		matrix[i][0:len(j)] = j
	
	if cutoff==0.0:
		cut = np.shape(matrix)[1]-2
		print("Compute the ensemble covariance matrix with no cutoff...")
	else:
		cut = int(cutoff/dt)
		print("Compute the ensemble covariance matrix with a cutoff at ",cutoff,"s...")
	
	covariance_matrix = np.zeros((cut,cut))
	N_tracks = np.shape(matrix)[0]

	for n in range(0,cut):
		for m in range(0,cut):
			s1=0
			s2=0
			s3=0
			Nnonzeron=0
			Nnonzerom=0
			Nnonzeronm=0
			for i in range(0,N_tracks):
				if matrix[i][n]!=0.0 and matrix[i][m]!=0.0:
					Nnonzeronm+=1
				if matrix[i][n]!=0.0:
					Nnonzeron+=1
				if matrix[i][m]!=0.0:
					Nnonzerom+=1
				s1+=matrix[i][n]*matrix[i][m]
				s2+=matrix[i][n]
				s3+=matrix[i][m]
			covariance_matrix[n][m] = (Nnonzeronm/(Nnonzeronm-1))*(1/float(Nnonzeronm)*s1 - 1/float(Nnonzeron)*1/float(Nnonzerom)*s2*s3)
	
	timelag = np.linspace(dt,dt*cut,cut)
	if plot==True:	
		sns.heatmap(covariance_matrix,xticklabels=[round(t,2) for t in timelag],yticklabels=[round(t,2) for t in timelag],cmap="YlGnBu")
		plt.xlabel(r'$n \Delta t$ (s)')
		plt.ylabel(r'$m \Delta t$ (s)')
		fix_heatmap()
		plt.show()
	print("Done. The ensemble covariance matrix has been generated from the set of MSD tracks.")
	return(covariance_matrix,timelag)
	
#######################################################
############## MICHALET LINEAR MODEL ##################
#######################################################

def LINEAR_MSD(MSD_Series,D,sigma,dt=0.05,cutoff=0.05,plot=False):
	"""Takes a series of MSD tracks of different lengths and simulates the linear model, assuming isotropic Brownian motion and that the central limit theorem is satisfied for all track lengths.  
	
	Parameters
	----------
	MSD_Series : list of lists/arrays
	    series of T-MSDs associated to each tracks
	D : float
	    theoretical diffusion coefficient
	sigma : float
	    theoretical localization uncertainty
	dt : float
	    time step between each T-MSD points. Default is dt=0.05 s
	cutoff : float
	    time lag at which the MSD will be cut
	plot : bool
	    plot the modelled MSD and the associated variance
	
	Returns
	-------
	mmsd_th
	   The modelled linear MSD
	mvar_th
	   The modelled variance for the MSD
	timelag
	    The associated time lag array
	"""
	print("Reading MSD tracklengths...")
	tracklengths = [len(msd)+1 for msd in MSD_Series] #the MSDs are shorter than the trajectories by one time lag
	
	#Michalet parameters for the localization uncertainty
	alpha = 4*D*dt
	epsilon = 4*sigma**2
	x = float(epsilon / alpha)

	cut = int(cutoff/dt)
	#Define linear MSD and timelag
	N = len(max(MSD_Series,key = lambda x: len(x)))+1
	N_T = len(MSD_Series)
	
	print("Generate linear MSD with D = ",round(D,4)," and epsilon = ",round(epsilon,3),"...")
	timelag = np.linspace(1*dt,(N-1)*dt,N-1)
	MSD = [4*D*t + epsilon for t in timelag]

	#Compute the variance for this distribution of tracklengths
	
	def f(n,N,x):
		"""
		Intermediate step
		This expression is derived by Michalet (2010)
		"""
		K = N-n
		if n<=K:
			fminus = n*(4*pow(n,2)*K + 2*K - pow(n,3) + n)/6/pow(K,2) + (2*n*x + (1+(1 - n/K)/2)*pow(x,2))/K
			return(1/fminus)
		else:
			fplus = (6*pow(n,2)*K - 4*n*pow(K,2) + 4*n + pow(K,3) - K)/6/K + (2*n*x + pow(x,2))/K
			return(1/fplus)
	
	
	variance_ensemble=[]
	print("Computing the the theoretical variance at each time lag for each MSD tracklength...")
	for N in tracklengths:
		variance_per_msd = []
		for n in range(0,N-1):
			variance_per_msd.append(alpha**2 / f(n+1,N,x)) #compute variance at each time lag for a MSD of length N-1 (time-variance)
		variance_ensemble.append(variance_per_msd) #build an ensemble of variance arrays
	
	#Generate a square matrix for the ensemble of the variances
	Nmax = len(max(MSD_Series,key = lambda x: len(x)))+1
	matrix = np.zeros([N_T,Nmax])
	for i,j in enumerate(variance_ensemble):
		matrix[i][0:len(j)] = j

	print("Performing the ensemble average of the variances associated to each MSD track...")
	variance = []
	# Perform the ensemble average for the variance
	for k in range(0,Nmax-1):
		s=0
		N_nonzero=0
		for l in range(0,N_T):
			if matrix[l][k]!=0.0:
				N_nonzero+=1
			s+=matrix[l][k]
		variance.append(1/float(N_nonzero)*s)
	
	if plot==True:	
		plt.plot(timelag,MSD,color='k')
		plt.fill_between(timelag,[msd+np.sqrt(var) for msd,var in zip(MSD,variance)], [msd-np.sqrt(var) for msd,var in zip(MSD,variance)],color='gray',alpha=0.1)
		plt.vlines(cutoff,plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyles='dashed',colors='k')
		plt.xlabel('Time lag (s)')
		plt.ylabel(r'MSD ($\mu$m$^2$)')
		plt.show()

	if cutoff>0.05:
		print("The MSD, variance and time lag arrays are cut at ",cutoff,'s...')
		MSD,variance,timelag = MSD[:cut],variance[:cut],timelag[:cut]
	print("Done. The linear model for the MSD and its variance have been generated.")
	return(MSD,variance,timelag)

def COVARIANCE_LINEAR_MSD(MSD_Series,D,sigma,dt=0.05,cutoff=0.0,plot=False):
	"""Takes a series of MSD of different lengths and simulates the theoretical covariance map, assuming isotropic Brownian motion and that the central limit theorem is satisfied for all track lengths. 
	
	Parameters
	----------
	MSD_Series : list of lists/arrays
	    series of T-MSDs associated to each tracks
	D : float
	    theoretical diffusion coefficient
	sigma : float
	    theoretical localization uncertainty
	dt : float
	    time step between each T-MSD points. Default is dt=0.05 s.
	cutoff : float
	    time lag at which the covariance matrix should be cut (because there not enough tracks to have a good average, or the computing time might be too long.)
	plot : bool
	    plots the covariance map
	
	Returns
	-------
	covariance_matrix
	   The ensemble covariance of the T-MSDs
	timelag
	    The associated time lag array
	"""
	print("Reading MSD tracklengths...")
	tracklengths = [len(msd)+1 for msd in MSD_Series]
	
	#Michalet parameters for the localization uncertainty
	epsilon=4*sigma**2
	alpha=4*D*dt
	
	cut = int(cutoff/dt)
	N_T = len(tracklengths)
	
	def g(m,n,N,sigma,D,dt):
		"""
		Intermediate step
		This expression is derived by Michalet (2010)
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
			sigmanm=n/(6*K*P)*(4*pow(n,2)*K + 2*K - pow(n,3) + n + (m-n)*(6*n*P - 4*pow(n,2) - 2))*pow(alpha,2)+1/K*(2*n*alpha*epsilon+(1-n/(2*P))*pow(epsilon,2))
			return(sigmanm)
		else:
			sigmanm=1/(6*K)*(6*pow(n,2)*K - 4*n*pow(K,2)+pow(K,3)+4*n - K +(m-n)*((n+m)*(2*K+P)+2*n*P-3*pow(K,2)+1))*pow(alpha,2)+1/K*(2*n*alpha*epsilon+pow(epsilon,2)/2)
			return(sigmanm)
	
	print("Compute covariance map for each track length...")
	covariance_ensemble = []
	for N in tracklengths:
		covariance_per_msd = np.zeros((N-1,N-1))
		for n in range(0,N-1):
			for m in range(0,N-1):
				covariance_per_msd[n,m] = g(n+1,m+1,N,sigma,D,dt)
				covariance_per_msd[m,n] = covariance_per_msd[n,m]
		covariance_ensemble.append(covariance_per_msd)

	print("Build intermediary series of matrices cut to ",cutoff," s...")
	temp = np.zeros((N_T,cut,cut)) #(Nloops,N-1,N-1)
	for k in range(0,N_T):
		length = np.shape(covariance_ensemble[k][:][:])[0]
		if length<cut:
			nshape=length
		else:
			nshape=cut
		for n in range(0,nshape):
			for m in range(0,nshape):
				temp[k][n][m] = covariance_ensemble[k][n][m]
	
	print("Compute the ensemble average of the covariance matrices...")
	covariance_matrix = np.zeros((cut,cut))
	for n in range(0,cut):
		for m in range(0,cut):
			Nnonzero = 0
			s=0
			for k in range(0,N_T):
				if temp[k][n][m]!=0.0:
					Nnonzero+=1
				s+=temp[k][n][m]
			covariance_matrix[n][m]=s/float(Nnonzero)
	
	timelag = np.linspace(dt,dt*cut,cut)
	if plot==True:
		sns.heatmap(covariance_matrix,xticklabels=[round(t,2) for t in timelag],yticklabels=[round(t,2) for t in timelag],cmap="YlGnBu")
		plt.xlabel(r'$n \Delta t$ (s)')
		plt.ylabel(r'$m \Delta t$ (s)')
		fix_heatmap()
		plt.show()
	print("Done. The ensemble covariance matrix has been generated from the set of MSD tracks.")
	return(covariance_matrix,timelag)
	
def WLS_MSD_SLOPE_ERROR(D,sigma,variance,covariance,dt=0.05):
	"""The function computes the relative error on the slope of a weighted least square fit of the MSD characterized by a variance array and a covariance matrix (sigma_b/b in Michalet's paper). The variance / covariance can be determined in any way, as long as their sizes are compatible (list length k, matrix (k,k)). Estimates for D and sigma are required for the scaling.
	
	Parameters
	----------
	D : float
	    estimate for the diffusion coefficient
	sigma : float
	    estimate for the localization uncertainty
	dt : float
	    time step between each MSD point. Default is dt=0.05 s.
	variance : list or ndarray
	    variance array associated to the MSD studied
	covariance : 2D list or 2D ndarray
	    covariance associtaed to the MSD
	
	Returns
	-------
	error_on_slope
	   The relative error on the slope as computed by weigthed least-square on the MSD.
	fit_points
	    The associated array of number of fitting points (first 2,3,...,k points to perform the WLS fit)
	"""
	lencov = np.shape(covariance)[0]
	if len(variance)!=lencov:
		print("The size of your variance does not match with the shape of your covariance matrix. Abort.")
		return
	
	cut = len(variance)
	fit_points = [int(p) for p in np.linspace(2,cut-2,cut-3)]
	error_on_slope = []
	print("The error on the slope is computed for each number of fitting points, based on the provided variance and covariance. The method used is weighted least squares...")
	for p in fit_points:
		alpha = 4*D*dt
		epsilon = 4*sigma**2
		x = float(epsilon / alpha)
		sum0=0
		sum1=0
		sum2=0
		sumh0=0
		sumh1=0
		sumh2=0
		for i in range(1,p+1):
			fi = alpha**2 / variance[i-1]
			sum0+=fi
			sum1+=i*fi
			sum2+=i**2*fi
			h0=0
			h1=0
			for j in range(1,i):
				fj = alpha**2 / variance[j-1]
				gij = covariance[i-1][j-1]/alpha**2
				hij = fi*fj*gij
				h0+=hij
				h1+=j*hij
			sumh0+=h0
			sumh1+=i*h0 + h1
			sumh2+=i*h1
		delta = sum0*sum2 - sum1**2
		temp = 1/delta*(sum0 + 2*(sum1**2*sumh0 - sum0*sum1*sumh1 + sum0**2*sumh2)/delta)
		if temp<0.0:
			print("Warning! Your square of the relative error is negative for p = ",p," fitting points...")
		norm_sigmab = np.sqrt(temp)
		error_on_slope.append(norm_sigmab)
	print("Done. The relative error for each number of fitting points has been computed.")
	min_y = min(error_on_slope) 
	min_x = fit_points[error_on_slope.index(min_y)]
	print("The relative error on the slope is minimum for P = ",min_x,"fitting points.")
	return(error_on_slope,fit_points)
	
	
def WLS_MSD_INTERCEPT_ERROR(D,sigma,variance,covariance,dt=0.05):
	"""The function computes the relative error on the intercept of a weighted least square fit of the MSD characterized by a variance array and a covariance matrix (sigma_b/b in Michalet's paper). The variance / covariance can be determined in any way, as long as their sizes are compatible (list length k, matrix (k,k)). Estimates for D and sigma are required for the scaling.
	
	Parameters
	----------
	D : float
	    estimate for the diffusion coefficient
	sigma : float
	    estimate for the localization uncertainty
	dt : float
	    time step between each MSD point. Default is dt=0.05 s.
	variance : list or ndarray
	    variance array associated to the MSD studied
	covariance : 2D list or 2D ndarray
	    covariance associtaed to the MSD
	
	Returns
	-------
	error_on_intercept
	   The relative error on the intercept as computed by weigthed least-square on the MSD.
	fit_points
	    The associated array of number of fitting points (first 2,3,...,k points to perform the WLS fit)
	"""
	lencov = np.shape(covariance)[0]
	if len(variance)!=lencov:
		print("The size of your variance does not match with the shape of your covariance matrix. Abort.")
		return
	
	cut = len(variance)
	fit_points = [int(p) for p in np.linspace(2,cut-2,cut-3)]
	error_on_intercept = []
	print("The error on the intercept is computed for each number of fitting points, based on the provided variance and covariance. The method used is weighted least squares...")
	for p in fit_points:
		alpha = 4*D*dt
		epsilon = 4*sigma**2
		x = float(epsilon / alpha)
		sum0=0
		sum1=0
		sum2=0
		sumh0=0
		sumh1=0
		sumh2=0
		for i in range(1,p+1):
			fi = alpha**2 / variance[i-1]
			sum0+=fi
			sum1+=i*fi
			sum2+=i**2*fi
			h0=0
			h1=0
			for j in range(1,i):
				fj = alpha**2 / variance[j-1]
				gij = covariance[i-1][j-1]/alpha**2
				hij = fi*fj*gij
				h0+=hij
				h1+=j*hij
			sumh0+=h0
			sumh1+=i*h0 + h1
			sumh2+=i*h1
		delta = sum0*sum2 - sum1**2
		temp = 1/delta*(sum2 + 2*(sum2**2*sumh0 - sum1*sum2*sumh1 + sum1**2*sumh2)/delta)
		if temp<0.0:
			print("Warning! Your square of the relative error is negative for p = ",p," fitting points...")
		norm_sigmaa = np.sqrt(temp)/x
		error_on_intercept.append(norm_sigmaa)
	print("Done. The relative error for each number of fitting points has been computed.")
	min_y = min(error_on_intercept) 
	min_x = fit_points[error_on_intercept.index(min_y)]
	print("The relative error on the intercept is minimum for P = ",min_x,"fitting points.")
	return(error_on_intercept,fit_points)


def Michalet(MSD_Series,dt=0.05,cutoff=0.0,theoretical_estimate=False,experiment_name='Simulation'):
	"""This functions takes a series of MSD curves and perform an ensemble analysis over these curves. It computes the TE-MSD, the associated variance and covariance. It evaluates the error on the fit of the TE-MSD to obtain optimal values for D (diffusion coefficient) and sigma (localization uncertainty). Once a first estimate for D and sigma has been obtained, it performs the whole process again to ensure that D and sigma have converged and to calibrate the matching theoretical linear model. 
	
	Parameters
	----------
	MSD_Series : list of lists/arrays
	    series of T-MSDs associated to each tracks
	dt : float
	    time step between each MSD point. Default is dt=0.05 s.
	cutoff : float (multiple of dt)
	    maximum time lag at which the analysis is performed 
	theoretical_estimate : bool
	    if true, use the theoretical variance to estimate the error on D and sigma, instead of the "ensemble" one. Useful if there is not enough data to satisfy the central limit theorem. 
	output : str
	    name for the output filename that will be stored in folder Plots/ as michalet_<output>.pdf
	   
	
	Returns
	-------
	error_on_intercept
	   The relative error on the intercept as computed by weigthed least-square on the MSD.
	fit_points
	    The associated array of number of fitting points (first 2,3,...,k points to perform the WLS fit)
	"""


	print("#############################################")
	print("############## PROGRAM MICHALET #############")
	print("#############################################")
	
	#Set any initial value, they will be adjusted...
	sigma=0.3
	D = 0.07
	epsilon = 4*sigma**2
	alpha = 4*D*dt
	
	if cutoff==0.0:
		print("No cutoff has been defined. Setting the cut at the maximum time lag.")
		cutoff = len(max(MSD_Series,key = lambda x: len(x)))*dt
	
	mmsd,mvar,timelag = TE_MSD(MSD_Series,cutoff=cutoff,plot=False,display_ntracks=True,dt=dt)
	covar,timelag = COVARIANCE(MSD_Series,cutoff=cutoff,dt=dt)
	
	for iteration in range(2):
		if iteration==0:
			print("Initial run to determine the best theoretical values for D and sigma.")
		if iteration==1:
			print("Second run with accurate values for D and sigma.")

		mmsd_th,mvar_th,timelag = LINEAR_MSD(MSD_Series,D,sigma,cutoff=cutoff,dt=dt)
		covar_th,timelag = COVARIANCE_LINEAR_MSD(MSD_Series,D,sigma,cutoff=cutoff,dt=dt)
		
		error_slope_exp,p_array = WLS_MSD_SLOPE_ERROR(D,sigma,mvar,covar)
		error_slope_th,p_array = WLS_MSD_SLOPE_ERROR(D,sigma,mvar_th,covar_th)

		min_y = min(error_slope_exp) 
		min_x = p_array[error_slope_exp.index(min_y)]
		
		if iteration==1 and theoretical_estimate==True:
			min_y = min(error_slope_th) 
			min_x = p_array[error_slope_th.index(min_y)]

		if iteration==0:
			print("It is estimated that the lowest error will be for a number of fitting points P = ",min_x," for which the relative error sigma_b / b = ",min_y)
		
		#Perform the fit of the MSD using the optimal number of points, and recovering an estimate for the slope
		timelag = np.array(timelag)
		x = timelag.reshape((-1, 1))
		y = mmsd
		ctffb = int(min_x)
		model = LinearRegression().fit(x[:ctffb], y[:ctffb])
		r_sq = model.score(x[:ctffb], y[:ctffb])
		fit_b = model.predict(x)

		# D is the slope / 4 and sigma_b / b = sigma_D / D...
		Dvalue = round(model.coef_[0]/4,4) 
		Dvalue_error = round(min_y*model.coef_[0]/4,4)
		print("D = ",Dvalue," +- ",Dvalue_error)
		
		error_intercept_exp,p_array = WLS_MSD_INTERCEPT_ERROR(D,sigma,mvar,covar)
		error_intercept_th,p_array = WLS_MSD_INTERCEPT_ERROR(D,sigma,mvar_th,covar_th)

		min_y = min(error_intercept_exp) 
		min_x = p_array[error_intercept_exp.index(min_y)]

		if iteration==1 and theoretical_estimate==True:
			min_y = min(error_intercept_th) 
			min_x = p_array[error_intercept_th.index(min_y)]

		if iteration==0:
			print("It is estimated that the lowest error will be when the number of fitting points P = ",min_x," for which the relative error sigma_a/a = ",min_y)

		#Perform the fit of the MSD using the optimal number of points, and recovering an estimate for the intercept
		timelag = np.array(timelag)
		xa = timelag.reshape((-1, 1))
		ya = mmsd
		ctffa = int(min_x)
		model = LinearRegression().fit(x[:ctffa], y[:ctffa])
		r_sq = model.score(x[:ctffa], y[:ctffa])
		fit_a = model.predict(x)

		# sigma is SQRT(epsilon/4) and epsilon is the intercept...
		loc_sigma = round(1/2*np.sqrt(model.intercept_+model.coef_[0]*0.05/3),4)
		loc_sigma_error = round(min_y/2*model.intercept_,4)
		print("sigma = ",loc_sigma,"+-",loc_sigma_error)

		D = Dvalue
		if np.isnan(loc_sigma)==True:
			print("sigma is nan. The square was probably negative. Replace with value arbitrarily close to zero.")
			sigma = 0.0000001
		else:
			sigma = loc_sigma

		epsilon = 4*sigma**2
		alpha = 4*D*dt
		if iteration==1:
			print("Done.")
			print("#############################################")
			print("####### GENERATING ALL OF THE PLOTS #########")
			print("#############################################")
			
	############## GLOBAL PLOT ###################################"

	fig = plt.figure(figsize=(16, 9))
	grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.5)
	msd_plot = fig.add_subplot(grid[0:2,0:2])
	cov_exp = fig.add_subplot(grid[2,0])
	cov_th  = fig.add_subplot(grid[2,1])
	frames_hist = fig.add_subplot(grid[0,2:4])
	slope_error_plot = fig.add_subplot(grid[1,2])
	intercept_error_plot = fig.add_subplot(grid[1,3])
	optimal_fit_D = fig.add_subplot(grid[2,2])
	optimal_fit_sigma = fig.add_subplot(grid[2,3])

	cexp = "#3e66b5"
	cth  = "#cf5c50"
	msize = 3.0

	######## MSD SUBPLOT #######################
	msd_plot.spines["top"].set_visible(False)  
	msd_plot.spines["right"].set_visible(False)
	msd_plot.get_xaxis().tick_bottom()  
	msd_plot.get_yaxis().tick_left()
	msd_plot.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(mmsd,mvar)],[a + np.sqrt(b) for a,b in zip(mmsd,mvar)], color="#b3d1ff",alpha=0.5)
	msd_plot.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(mmsd_th,mvar_th)],[a + np.sqrt(b) for a,b in zip(mmsd_th,mvar_th)], color="#febab3",alpha=0.5) 
	msd_plot.plot(timelag,mmsd,color=cexp, lw=2,label="Experimental MSD $\pm \sigma $")
	msd_plot.plot(timelag,mmsd_th,color=cth, lw=2,label="Theoretical MSD $\pm \sigma $")
	msd_plot.set_ylabel(r"Mean square displacement ($\mu$m$^2$)",fontsize=10)
	msd_plot.set_xlabel('Time lag (s)',fontsize=10)
	msd_plot.legend(loc="upper left",fontsize=10)

	########## HISTOGRAM ########################"
	N_array = [len(msd)+1 for msd in MSD_Series]
	frames_hist.spines["top"].set_visible(False)  
	frames_hist.spines["right"].set_visible(False)
	frames_hist.get_xaxis().tick_bottom()  
	frames_hist.get_yaxis().tick_left()
	hist_array = [n*dt for n in N_array]
	frames_hist.hist(hist_array,color=cexp)
	frames_hist.axvline(cutoff, 0, max(hist_array),color=cth)
	frames_hist.set_xlabel('Track duration (s)',fontsize=10)
	frames_hist.set_ylabel('Number of tracks',fontsize=10)

	######## COVARIANCE #############################

	im1 = cov_exp.pcolormesh(timelag,timelag,covar,cmap="Blues")
	cov_exp.set_xlabel(r'$n \Delta t$')
	cov_exp.set_ylabel(r'$m \Delta t$')
	cbar = plt.colorbar(im1,ax=cov_exp)
	#plt.title('Map of experimental covariance values')

	im2 = cov_th.pcolormesh(timelag,timelag,covar_th,cmap="Reds")
	#cov_th.colorbar()
	cov_th.set_xticks([])
	cov_th.set_yticks([])
	cov_th.set_xlabel(r'$n \Delta t$')
	cov_th.set_ylabel(r'$m \Delta t$')
	cbar = plt.colorbar(im2,ax=cov_th)
	#cov_th.set_title('Map of theoretical covariance values')


	############## ERRORS ###################################
	slope_error_plot.spines["top"].set_visible(False)  
	slope_error_plot.spines["right"].set_visible(False)
	slope_error_plot.get_xaxis().tick_bottom()  
	slope_error_plot.get_yaxis().tick_left()
	slope_error_plot.loglog(p_array,error_slope_th,label="Theory",color=cth)
	slope_error_plot.loglog(p_array,error_slope_exp,"-x",label="Experiment",color=cexp,ms=msize)
	slope_error_plot.set_xlabel('Number of fitting points $P$')
	slope_error_plot.set_ylabel(r'$\sigma_b / b$')
	slope_error_plot.legend(fontsize=8)

	intercept_error_plot.spines["top"].set_visible(False)  
	intercept_error_plot.spines["right"].set_visible(False)
	intercept_error_plot.get_xaxis().tick_bottom()  
	intercept_error_plot.get_yaxis().tick_left()
	intercept_error_plot.loglog(p_array,error_intercept_th,label="Theory",color=cth)
	intercept_error_plot.loglog(p_array,error_intercept_exp,"-x",label="Experiment",color=cexp,ms=msize)
	intercept_error_plot.set_xlabel('Number of fitting points $P$')
	intercept_error_plot.set_ylabel(r'$\sigma_a / a$')
	intercept_error_plot.legend(fontsize=8)

	############" LINEAR FITS ######################################
	optimal_fit_D.spines["top"].set_visible(False)  
	optimal_fit_D.spines["right"].set_visible(False)
	optimal_fit_D.get_xaxis().tick_bottom()  
	optimal_fit_D.get_yaxis().tick_left()
	optimal_fit_D.plot(x, fit_b,"-x",label=r'$P = $'+str(ctffb),color="purple",ms=msize)
	optimal_fit_D.plot(timelag,mmsd,"-x",label="Exp.",color=cexp,ms=msize)
	optimal_fit_D.plot(timelag,mmsd_th,label="Th.",color=cth)
	optimal_fit_D.legend(loc="upper left",fontsize=8)
	optimal_fit_D.set_xlabel('Timelag (s)')
	optimal_fit_D.set_ylabel(r'MSD ($\mu$m$^2$)')
	ymin, ymax = optimal_fit_D.get_ylim()
	optimal_fit_D.text(max(timelag)/5,ymin,r'$D = ($'+str(Dvalue)+"$\pm$"+str(Dvalue_error)+") $\mu m^2 / s$",color="purple",fontsize=8)

	optimal_fit_sigma.spines["top"].set_visible(False)  
	optimal_fit_sigma.spines["right"].set_visible(False)
	optimal_fit_sigma.get_xaxis().tick_bottom()  
	optimal_fit_sigma.get_yaxis().tick_left()
	optimal_fit_sigma.plot(x, fit_a,"-x",label=r'$P = $'+str(ctffa),color="purple",ms=msize)
	optimal_fit_sigma.plot(timelag,mmsd,"x-",label="Exp.",color=cexp,ms=msize)
	optimal_fit_sigma.plot(timelag,mmsd_th,label="Th.",color=cth)
	optimal_fit_sigma.legend(loc="upper left",fontsize=8)
	optimal_fit_sigma.set_xlabel('Timelag (s)')
	optimal_fit_sigma.set_ylabel(r'MSD ($\mu$m$^2$)')
	ymin, ymax = optimal_fit_sigma.get_ylim()
	optimal_fit_sigma.text(max(timelag)/3,ymin,r'$\sigma_0 = ($'+str(loc_sigma)+"$\pm$"+str(loc_sigma_error)+") $\mu$m$^2$",color="purple",fontsize=8)

	plt.show()
	
	width = 469.75502
	fig= plt.figure(figsize=set_size(width, fraction=1))
	grid = plt.GridSpec(3, 4, wspace=1.2, hspace=0.9,bottom=0.2)
	msd_plot = fig.add_subplot(grid[0:2,0:2])
	cov_exp = fig.add_subplot(grid[2,0])
	cov_th  = fig.add_subplot(grid[2,1])
	slope_error_plot = fig.add_subplot(grid[0,2:4])
	intercept_error_plot = fig.add_subplot(grid[1,2:4])
	optimal_fit_D = fig.add_subplot(grid[2,2])
	optimal_fit_sigma = fig.add_subplot(grid[2,3])

	cexp = "tab:blue"
	cth  = "tab:red"
	msize = 2

	######## MSD SUBPLOT #######################
	msd_plot.spines["top"].set_visible(False)  
	msd_plot.spines["right"].set_visible(False)
	msd_plot.get_xaxis().tick_bottom()  
	msd_plot.get_yaxis().tick_left()
	msd_plot.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(mmsd,mvar)],[a + np.sqrt(b) for a,b in zip(mmsd,mvar)], color="#b3d1ff",alpha=0.5)
	msd_plot.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(mmsd_th,mvar_th)],[a + np.sqrt(b) for a,b in zip(mmsd_th,mvar_th)], color="#febab3",alpha=0.5) 
	msd_plot.plot(timelag,mmsd,color=cexp, lw=2,label=experiment_name)
	msd_plot.plot(timelag,mmsd_th,color=cth, lw=2,label="Theory")
	msd_plot.set_ylabel(r"TE-MSD ($\mu$m$^2$)")
	msd_plot.set_xlabel('Time lag (s)')
	msd_plot.legend(loc="upper left")

	########## HISTOGRAM ########################"
	N_array = [len(msd)+1 for msd in MSD_Series]
	#hist.spines["top"].set_visible(False)  
	#hist.spines["right"].set_visible(False)
	#hist.get_xaxis().tick_bottom()  
	#hist.get_yaxis().tick_left()
	#hist_array = [n*dt for n in N_array]
	#hist.hist(hist_array,color=cexp, bins=int(len(N_array)))
	#hist.axvline(cutoff/dt, 0, max(hist_array),color=cth)
	#hist.set_xlabel('Track duration (s)',fontsize=10)
	#hist.set_ylabel('Number of tracks',fontsize=10)

	######## COVARIANCE #############################

	im1 = cov_exp.pcolormesh(timelag,timelag,covar,cmap="Blues")
	cov_exp.set_xlabel(r'$n \Delta t$')
	cov_exp.set_ylabel(r'$m \Delta t$')
	cov_exp.set_xticks([])
	cov_exp.set_yticks([])
	cbar = plt.colorbar(im1,ax=cov_exp)
	cbar.ax.tick_params(labelsize=8)
	#plt.title('Map of experimental covariance values')
	#cov_exp.text(0,-1.5,'Covariance maps')

	im2 = cov_th.pcolormesh(timelag,timelag,covar_th,cmap="Reds")
	#cov_th.colorbar()
	cov_th.set_xticks([])
	cov_th.set_yticks([])
	cov_th.set_xlabel(r'$n \Delta t$')
	cov_th.set_ylabel(r'$m \Delta t$')
	cbar = plt.colorbar(im2,ax=cov_th)
	cbar.ax.tick_params(labelsize=8)

	############## ERRORS ###################################
	slope_error_plot.spines["top"].set_visible(False)  
	slope_error_plot.spines["right"].set_visible(False)
	slope_error_plot.loglog(p_array,error_slope_th,label="Theory",color=cth)
	slope_error_plot.loglog(p_array,error_slope_exp,"-x",label="Experiment",color=cexp,ms=msize)
	slope_error_plot.set_xlabel('Number of fitting points')
	slope_error_plot.set_ylabel(r'$\sigma_b / b$')
	slope_error_plot.tick_params(axis='both', which='major', labelsize=8)
	slope_error_plot.tick_params(axis='both', which='minor', labelsize=8)
	slope_error_plot.xaxis.set_minor_formatter(mticker.ScalarFormatter())
	slope_error_plot.yaxis.set_minor_formatter(mticker.ScalarFormatter())
	slope_error_plot.xaxis.set_major_formatter(mticker.ScalarFormatter())
	slope_error_plot.yaxis.set_major_formatter(mticker.ScalarFormatter())
	slope_error_plot.yaxis.set_minor_locator(plt.MaxNLocator(2))
	slope_error_plot.xaxis.set_minor_locator(plt.MaxNLocator(5))
	#sigmab_err.legend(fontsize=8)

	intercept_error_plot.spines["top"].set_visible(False)  
	intercept_error_plot.spines["right"].set_visible(False)
	#sigmaa_err.get_xaxis().tick_bottom()  
	#sigmaa_err.get_yaxis().tick_left()
	intercept_error_plot.loglog(p_array,error_intercept_th,label="Theory",color=cth)
	intercept_error_plot.loglog(p_array,error_intercept_exp,"-x",label="Experiment",color=cexp,ms=msize)
	intercept_error_plot.set_xlabel('Number of fitting points')
	intercept_error_plot.set_ylabel(r'$\sigma_a / a$')
	intercept_error_plot.tick_params(axis='both', which='major', labelsize=8)
	intercept_error_plot.tick_params(axis='both', which='minor', labelsize=8)
	intercept_error_plot.xaxis.set_minor_formatter(mticker.ScalarFormatter())
	intercept_error_plot.yaxis.set_minor_formatter(mticker.ScalarFormatter())
	intercept_error_plot.xaxis.set_major_formatter(mticker.ScalarFormatter())
	intercept_error_plot.yaxis.set_major_formatter(mticker.ScalarFormatter())
	intercept_error_plot.yaxis.set_minor_locator(plt.MaxNLocator(2))
	intercept_error_plot.xaxis.set_minor_locator(plt.MaxNLocator(5))
	#sigmaa_err.set_xticklabels([2,10])
	#sigmaa_err.legend(fontsize=8)

	############" LINEAR FITS ######################################
	optimal_fit_D.spines["top"].set_visible(False)  
	optimal_fit_D.spines["right"].set_visible(False)
	optimal_fit_D.get_xaxis().tick_bottom()  
	optimal_fit_D.get_yaxis().tick_left()
	optimal_fit_D.plot(x, fit_b,"--",label=r'$P = $'+str(ctffb),color="tab:purple",linewidth=1)
	optimal_fit_D.plot(timelag,mmsd,"-x",color=cexp,ms=msize,linewidth=0.5)
	#D_plot.plot(timelag,mmsd_th,label="Th.",color=cth)
	optimal_fit_D.legend(loc="upper left",fontsize=6)
	optimal_fit_D.set_xlabel('Time lag (s)',fontsize=8)
	optimal_fit_D.set_ylabel(r'TE-MSD ($\mu$m$^2$)',fontsize=8)
	optimal_fit_D.yaxis.set_label_position("left")
	optimal_fit_D.set_yticklabels([])
	optimal_fit_D.tick_params(axis='both', which='major', labelsize=8)
	optimal_fit_D.tick_params(axis='both', which='minor', labelsize=8)

	#D_plot.text(max(timelag)/5,max(mmsd)/10,r'$D = ($'+str(Dvalue)+"$\pm$"+str(Dvalue_error)+") $\mu m^2 / s$",color="k",fontsize=3)

	optimal_fit_sigma.spines["top"].set_visible(False)  
	optimal_fit_sigma.spines["right"].set_visible(False)
	optimal_fit_sigma.get_xaxis().tick_bottom()  
	optimal_fit_sigma.get_yaxis().tick_left()
	optimal_fit_sigma.plot(x, fit_a,"--",label=r'$P = $'+str(ctffa),color="tab:purple",linewidth=1)
	optimal_fit_sigma.plot(timelag,mmsd,"x-",color=cexp,ms=msize,linewidth=0.5)
	#loc_unc_plot.plot(timelag,mmsd_th,label="Th.",color=cth)
	optimal_fit_sigma.legend(loc="upper left",fontsize=6)
	optimal_fit_sigma.set_xlabel('Time lag (s)',fontsize=8)
	optimal_fit_sigma.tick_params(axis='both', which='major', labelsize=8)
	optimal_fit_sigma.tick_params(axis='both', which='minor', labelsize=8)
	#loc_unc_plot.set_ylabel(r'MSD ($\mu$m$^2$)')
	#loc_unc_plot.yaxis.set_label_position("right")
	#loc_unc_plot.yaxis.tick_right()
	#loc_unc_plot.text(max(timelag)/3,max(mmsd)/10,r'$\sigma_0 = ($'+str(loc_sigma)+"$\pm$"+str(loc_sigma_error)+") $\mu$m$^2$",color="k",fontsize=3)

	plt.suptitle(r'$D =$ ('+str(Dvalue)+"$\pm$"+str(Dvalue_error)+") $\mu$m$^2/$s,"+' $\sigma = ($'+str(loc_sigma)+"$\pm$"+str(loc_sigma_error)+") $\mu$m")
	plt.tight_layout()
	
	
	return(Dvalue,Dvalue_error,loc_sigma,loc_sigma_error)

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
		binsize = int(nloop/50)
		hist,bin_edge = np.histogram(stat_array,bins=binsize)
		ax = plt.subplot(111)
		ax.hist(stat_array,bins=binsize,alpha=0.6)
		ax.vlines(stat0,0,max(hist),color='r')
		ax.set_xlabel(r'KS statistic $D^*$')
		ax.set_ylabel('#')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_title('Statistical distribution for the K-S statistic\n with $D_0$ = '+str(round(stat0,3))+" and a p-value = "+str(round(pvalue,5))+"\n")
		plt.show()
	return(stat0,pvalue)



#######################################################
############## PLOT FUNCTIONS #########################
#######################################################

def set_size(width, fraction=1):
	""" Set figure dimensions to avoid scaling in LaTeX.
	This function was developed at https://jwalton.info/ by Jack Walton

	Parameters
	----------
	width: float
	    Document textwidth or columnwidth in pts
	fraction: float, optional
	    Fraction of the width which you wish the figure to occupy

	Returns
	-------
	fig_dim: tuple
	    Dimensions of figure in inches
	"""
	# Width of figure (in pts)
	fig_width_pt = width * fraction

	# Convert from pt to inches
	inches_per_pt = 1 / 72.27

	# Golden ratio to set aesthetic figure height
	# https://disq.us/p/2940ij3
	golden_ratio = (5**.5 - 1) / 2

	# Figure width in inches
	fig_width_in = fig_width_pt * inches_per_pt
	# Figure height in inches
	fig_height_in = fig_width_in * golden_ratio

	fig_dim = (fig_width_in, fig_height_in)

	return(fig_dim)

def fix_heatmap():
	"""
	This function fixes seaborn heat maps. To be placed before plt.show()
	"""
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values