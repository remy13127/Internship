import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import data_pool,NormMSDSlopeError,NormMSDInterceptError,NormMSDSlopeError_exp,NormMSDInterceptError_exp,fix_heatmap
from sklearn.linear_model import LinearRegression


def process_msd_series(MSD_Series):
	matrix = np.zeros([len(MSD_Series),len(max(MSD_Series,key = lambda x: len(x)))])
	for i,j in enumerate(MSD_Series):
		matrix[i][0:len(j)] = j
	Narray = [len(msd)+1 for msd in MSD_Series]
	return(matrix,Narray)

def ensemble_average(matrix):
	N=np.shape(matrix)[1]+1
	N_T=np.shape(matrix)[0]

	VARIANCE = []
	MSD = []

	for k in range(0,N-1):
		s1=0
		s2=0
		N_nonzero=0
		for l in range(0,N_T):
			if matrix[l][k]!=0.0:
				N_nonzero+=1
			s1+=matrix[l][k]**2
			s2+=matrix[l][k]
		VARIANCE.append(1/float(N_nonzero)*s1 - (1/float(N_nonzero)*s2)**2)
		MSD.append(1/float(N_nonzero)*s2)
	return(MSD,VARIANCE)


def mean_msd(MSD_Series,dt=0.05,cutoff=0.0,show_cutoff=False):
	"""
	This function takes a list of MSDs and returns the ensemble MSD and the associated variance. 
	Input: <list of lists> or <list of ndarray>
	Returns: ensemble MSD<ndarray>, ensemble variance <ndarray> and timelag<ndarray> of the length of the longest MSD in the initial sample. 
	"""
	matrix = np.zeros([len(MSD_Series),len(max(MSD_Series,key = lambda x: len(x)))])
	for i,j in enumerate(MSD_Series):
		matrix[i][0:len(j)] = j

	MMSD = []
	MVAR = []

	for k in range(0,np.shape(matrix)[1]):
		s1=0
		s2=0
		N_nonzero=0
		for l in range(0,np.shape(matrix)[0]):
			if matrix[l][k]!=0.0:
				N_nonzero+=1
			s1+=matrix[l][k]**2
			s2+=matrix[l][k]
		MVAR.append(1/float(N_nonzero)*s1 - (1/float(N_nonzero)*s2)**2)
		MMSD.append(1/float(N_nonzero)*s2)	
	
	TIMELAG = np.linspace(1*dt,len(MMSD)*dt,len(MMSD))
	
	plt.plot(TIMELAG,MMSD,color='k')
	plt.fill_between(TIMELAG,[a+np.sqrt(b) for a,b in zip(MMSD,MVAR)], [a-np.sqrt(b) for a,b in zip(MMSD,MVAR)],color='gray',alpha=0.1)
	if show_cutoff==True:
		plt.vlines(cutoff,plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyles='dashed',colors='k')
	else:
		plt.vlines(timelag[-1],plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],linestyles='dashed',colors='k')
	plt.xlabel('Time lag (s)')
	plt.ylabel(r'Ensemble average of the MSD ($\mu$m$^2$)')
	plt.show()
	
	if cutoff!=0.0:
		cut = int(cutoff/dt)
		MMSD,MVAR,TIMELAG = MMSD[:cut],MVAR[:cut],TIMELAG[:cut]
	return(MMSD,MVAR,TIMELAG)
	
def covariance_msd(MSD_Series,dt=0.05,cutoff=0.0):

	matrix = np.zeros([len(MSD_Series),len(max(MSD_Series,key = lambda x: len(x)))])
	for i,j in enumerate(MSD_Series):
		matrix[i][0:len(j)] = j
	
	cut = int(cutoff/dt)	
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
			covariance_matrix[n][m] = 1/float(Nnonzeronm)*s1 - 1/float(Nnonzeron)*1/float(Nnonzerom)*s2*s3
	return(covariance_matrix)
	

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
        

def theoretical_mean_msd(MSD_Series,D,sigma,dt=0.05,cutoff=0.0):
	''' This program computes the theoretical ensemble average for the MSD and ensemble variance for free diffusion. 
	N: total number of time steps
	sigma: localization uncertainty due to noise (identical in X and Y)
	D: diffusion coefficient (measured)
	t: total duration of simulation
	'''
	
	FRAME_LENGTH = [len(msd)+1 for msd in MSD_Series]
	alpha = 4*D*dt
	epsilon = 4*sigma**2
	x = epsilon / alpha

	
	cut = int(cutoff/dt)
	timelag = np.linspace(1*dt,len(max(MSD_Series,key = lambda x: len(x)))*dt,len(max(MSD_Series,key = lambda x: len(x))))
	mmsd_th = [4*D*ndt + epsilon for ndt in timelag]
	
	variance_ensemble=[]
	
	for N in FRAME_LENGTH:
		variance_per_msd = []
		for n in range(0,N-1):
			variance_per_msd.append(alpha**2 / f(n+1,N,x))
		variance_ensemble.append(variance_per_msd)

	matrix = np.zeros([len(variance_ensemble),len(max(variance_ensemble,key = lambda x: len(x)))])
	for i,j in enumerate(variance_ensemble):
		matrix[i][0:len(j)] = j

	mvar_th = []
	N=np.shape(matrix)[1]+1
	N_T=np.shape(matrix)[0]
	
	for k in range(0,N-1):
		s=0
		N_nonzero=0
		for l in range(0,N_T):
			if matrix[l][k]!=0.0:
				N_nonzero+=1
			s+=matrix[l][k]
		mvar_th.append(1/float(N_nonzero)*s)

	if cutoff!=0.0:
		mmsd_th,mvar_th = mmsd_th[:cut],mvar_th[:cut]
	
	return(mmsd_th,mvar_th)

def g(m,n,N,sigma,D,dt):
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



def theoretical_covariance(MSD_Series,D,sigma,dt=0.05,cutoff=0.0):
	"""Cross correlation variance sigma_{nm}^2 
	N: total number of time steps
	sigma: localization uncertainty
	D: Diffusion coefficient
	t: duration of experiment / simulation 
	"""
	epsilon=4*sigma**2
	alpha=4*D*dt
	covariance_ensemble = []
	FRAME_LENGTH = [len(msd)+1 for msd in MSD_Series]
	
	for N in FRAME_LENGTH:
		covariance_per_msd = np.zeros((N-1,N-1))
		for n in range(0,N-1):
			for m in range(0,N-1):
				covariance_per_msd[n,m] = g(n+1,m+1,N,sigma,D,dt)
				covariance_per_msd[m,n] = covariance_per_msd[n,m]
		covariance_ensemble.append(covariance_per_msd)
	
	N_tracks = len(FRAME_LENGTH)

	cut = int(cutoff/dt)
	temp = np.zeros((N_tracks,cut,cut)) #(Nloops,N-1,N-1)
	for k in range(0,N_tracks):
		length = np.shape(covariance_ensemble[k][:][:])[0]
		if length<cut:
			nshape=length
		else:
			nshape=cut
		for n in range(0,nshape):
			for m in range(0,nshape):
				temp[k][n][m] = covariance_ensemble[k][n][m]

	covariance_matrix = np.zeros((cut,cut))
	for n in range(0,cut):
		for m in range(0,cut):
			Nnonzero = 0
			s=0
			for k in range(0,N_tracks):
				if temp[k][n][m]!=0.0:
					Nnonzero+=1
				s+=temp[k][n][m]
			covariance_matrix[n][m]=s/float(Nnonzero)
	return(covariance_matrix)


def Michalet(MSD_Series,dt=0.05,cutoff=0.0):
	sigma=0.3
	D = 0.07
	epsilon = 4*sigma**2
	alpha = 4*D*dt
	
	mmsd,mvar,timelag = mean_msd(MSD_Series,cutoff=cutoff,show_cutoff=True)
	covar = covariance_msd(MSD_Series,cutoff=cutoff)
	N=len(mmsd)+1
	
	for iteration in range(2):
		if iteration==0:
			print("Initial run to determine the best theoretical values for D and sigma.")
		if iteration==1:
			print("Second run with accurate values for D and sigma.")

		mmsd_th,mvar_th = theoretical_mean_msd(MSD_Series,D,sigma,cutoff=cutoff)
		covar_th = theoretical_covariance(MSD_Series,D,sigma,cutoff=cutoff)
		###########################################
		############### ERROR SLOPE ###############
		cut = int(cutoff/dt)
		Pmin_array = [int(p) for p in np.linspace(2,cut-2,cut-3)]
		sigmab_theory = []
		sigmab_exp_th = []
		sigmab_exp    = []

		for p in Pmin_array:
			sigmab_theory.append(NormMSDSlopeError(N,sigma,D,dt,p))
			sigmab_exp_th.append(NormMSDSlopeError_exp(N,sigma,D,dt,mvar_th,covar_th,p))
			sigmab_exp.append(NormMSDSlopeError_exp(N,sigma,D,dt,mvar,covar,p))

		min_y = min(sigmab_exp) 
		min_x = Pmin_array[sigmab_exp.index(min_y)]

		if iteration==0:
			print("It is estimated that the lowest error will be for a number of fitting points P = ",min_x," for which the relative error sigma/b = ",min_y)
		
		timelag = np.array(timelag)
		x = timelag.reshape((-1, 1))
		y = mmsd
		ctffb = int(min_x)
		model = LinearRegression().fit(x[:ctffb], y[:ctffb])
		r_sq = model.score(x[:ctffb], y[:ctffb])
		fit_b = model.predict(x)

		Dvalue = round(model.coef_[0]/4,4)
		print("D = ",Dvalue)
		Dvalue_error = round(min_y*model.coef_[0]/4,4)


		###########################################
		############### ERROR INTERCEPT ###############

		Pmin_array = [int(p) for p in np.linspace(2,cut-2,cut-3)]
		sigmaa_theory = []
		sigmaa_exp_th = []
		sigmaa_exp    = []
		for p in Pmin_array:
			sigmaa_theory.append(NormMSDInterceptError(N,sigma,D,dt,p))
			sigmaa_exp_th.append(NormMSDInterceptError_exp(N,sigma,D,dt,mvar_th,covar_th,p))
			sigmaa_exp.append(NormMSDInterceptError_exp(N,sigma,D,dt,mvar,covar,p))

		min_y = min(sigmaa_exp) 
		min_x = Pmin_array[sigmaa_exp.index(min_y)]

		if iteration==0:
			print("It is estimated that the lowest error will be when the number of fitting points P = ",min_x," for which the relative error sigmaa/a = ",min_y)

		timelag = np.array(timelag)
		xa = timelag.reshape((-1, 1))
		ya = mmsd
		ctffa = int(min_x)
		model = LinearRegression().fit(x[:ctffa], y[:ctffa])
		r_sq = model.score(x[:ctffa], y[:ctffa])
		fit_a = model.predict(x)

		loc_sigma = round(np.sqrt(model.intercept_/4),4)
		loc_sigma_error = round(min_y/2*model.intercept_,4)
		print("sigma = ",loc_sigma)

		D = Dvalue
		if np.isnan(loc_sigma)==True:
			print("sigma is nan. Replace with value close to zero.")
			sigma = 0.0000001
		else:
			sigma = loc_sigma

		epsilon = 4*sigma**2
		alpha = 4*D*dt
		if iteration==1:
			print("Done.")
			
	############## GLOBAL PLOT ###################################"

	fig = plt.figure(figsize=(16, 9))
	grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.5)
	msd_plot = fig.add_subplot(grid[0:2,0:2])
	cov_exp = fig.add_subplot(grid[2,0])
	cov_th  = fig.add_subplot(grid[2,1])
	hist = fig.add_subplot(grid[0,2:4])
	sigmab_err = fig.add_subplot(grid[1,2])
	sigmaa_err = fig.add_subplot(grid[1,3])
	D_plot = fig.add_subplot(grid[2,2])
	loc_unc_plot = fig.add_subplot(grid[2,3])

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
	hist.spines["top"].set_visible(False)  
	hist.spines["right"].set_visible(False)
	hist.get_xaxis().tick_bottom()  
	hist.get_yaxis().tick_left()
	hist_array = [n*dt for n in N_array]
	hist.hist(hist_array,color=cexp, bins=int(len(N_array)))
	hist.axvline(cutoff/dt, 0, max(hist_array),color=cth)
	hist.set_xlabel('Track duration (s)',fontsize=10)
	hist.set_ylabel('Number of tracks',fontsize=10)

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
	sigmab_err.spines["top"].set_visible(False)  
	sigmab_err.spines["right"].set_visible(False)
	sigmab_err.get_xaxis().tick_bottom()  
	sigmab_err.get_yaxis().tick_left()
	sigmab_err.loglog(Pmin_array,sigmab_exp_th,label="Theory",color=cth)
	sigmab_err.loglog(Pmin_array,sigmab_exp,"-x",label="Experiment",color=cexp,ms=msize)
	sigmab_err.set_xlabel('Number of fitting points $P$')
	sigmab_err.set_ylabel(r'$\sigma_b / b$')
	sigmab_err.legend(fontsize=8)

	sigmaa_err.spines["top"].set_visible(False)  
	sigmaa_err.spines["right"].set_visible(False)
	sigmaa_err.get_xaxis().tick_bottom()  
	sigmaa_err.get_yaxis().tick_left()
	sigmaa_err.loglog(Pmin_array,sigmaa_exp_th,label="Theory",color=cth)
	#sigmaa_err.loglog(Pmin_array,sigmaa_exp_th,".",label="Control_theory from matrices")
	sigmaa_err.loglog(Pmin_array,sigmaa_exp,"-x",label="Experiment",color=cexp,ms=msize)
	sigmaa_err.set_xlabel('Number of fitting points $P$')
	sigmaa_err.set_ylabel(r'$\sigma_a / a$')
	sigmaa_err.legend(fontsize=8)

	############" LINEAR FITS ######################################
	D_plot.spines["top"].set_visible(False)  
	D_plot.spines["right"].set_visible(False)
	D_plot.get_xaxis().tick_bottom()  
	D_plot.get_yaxis().tick_left()
	D_plot.plot(x, fit_b,"-x",label=r'$P = $'+str(ctffb),color="purple",ms=msize)
	D_plot.plot(timelag,mmsd,"-x",label="Exp.",color=cexp,ms=msize)
	D_plot.plot(timelag,mmsd_th,label="Th.",color=cth)
	D_plot.legend(loc="upper left",fontsize=8)
	D_plot.set_xlabel('Timelag (s)')
	D_plot.set_ylabel(r'MSD ($\mu$m$^2$)')
	D_plot.text(max(timelag)/5,max(mmsd)/10,r'$D = ($'+str(Dvalue)+"$\pm$"+str(Dvalue_error)+") $\mu m^2 / s$",color="purple",fontsize=8)

	loc_unc_plot.spines["top"].set_visible(False)  
	loc_unc_plot.spines["right"].set_visible(False)
	loc_unc_plot.get_xaxis().tick_bottom()  
	loc_unc_plot.get_yaxis().tick_left()
	loc_unc_plot.plot(x, fit_a,"-x",label=r'$P = $'+str(ctffa),color="purple",ms=msize)
	loc_unc_plot.plot(timelag,mmsd,"x-",label="Exp.",color=cexp,ms=msize)
	loc_unc_plot.plot(timelag,mmsd_th,label="Th.",color=cth)
	loc_unc_plot.legend(loc="upper left",fontsize=8)
	loc_unc_plot.set_xlabel('Timelag (s)')
	loc_unc_plot.set_ylabel(r'MSD ($\mu$m$^2$)')
	loc_unc_plot.text(max(timelag)/3,max(mmsd)/10,r'$\sigma_0 = ($'+str(loc_sigma)+"$\pm$"+str(loc_sigma_error)+") $\mu$m$^2$",color="purple",fontsize=8)

	plt.show()