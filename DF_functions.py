import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def mean_msd(MSD_Series,dt=0.05,cutoff=0.0):
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

