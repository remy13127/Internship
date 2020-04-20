import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dt', '-dt', help='Set dt for the track.')
parser.add_argument('-filename', '-filename', help='Name of the original .csv file.')
parser.add_argument('-clustercolor', '-clustercolor', help='Color of the chosen cluster.')

args = parser.parse_args()
dt = float(args.dt)
filename = args.filename
clustercolor = args.clustercolor


folder_name = filename[:-4]
output_folder = "Result/"+folder_name+'/'+clustercolor

os.system('mkdir '+str(output_folder))
os.system('cp filtered_msd.txt '+output_folder+'/.')
os.system('cp '+"Result/"+folder_name+'/data_characteristics.png'+' '+output_folder+'/.')

print("Reading data in filtered_msd.txt ...")
print("Compute MSD curves for each track in the data...")

data,N_array = MSD_tracks_from_file("filtered_msd.txt")

N = np.shape(data)[1]+1
D= 0.1603
sigma=0.0874
epsilon = 4*sigma**2
alpha = 4*D*dt
x = sigma**2 / (D*dt)
T = N*dt

##############" PLOT ALL DATA MSD #################
print("Use all tracks to compute an ensemble MSD and its associated variance...")
msd,var = average(data)


print("Plot ensemble MSD...")
timelag = np.linspace(1,N-1,N-1)


plt.figure(figsize=(8, 6))
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True,which='both')
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
plt.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(msd,var)],[a + np.sqrt(b) for a,b in zip(msd,var)], color="#3F5D7D") 
plt.plot(timelag,msd,color="white", lw=2)
plt.ylabel(r"Mean square displacement ($\mu$m$^2/$s)",fontsize=16)
plt.xlabel('Time Lag (n)',fontsize=16)
plt.show()

### SELECT CUTOFF #################################

cutoff = int(input("After how many time steps n do you want to cut the data? ")) # crop data 
if cutoff > N-1:
	print("Your cutoff is larger than the available data !")
else:
	print("You have set the cutoff to a time lag of ",str(cutoff)," steps...")

data = data[:,:cutoff]
timelag = np.linspace(1,N-1,N-1)[:cutoff]
var = var[:cutoff]
msd = msd[:cutoff]


##### PLOT MSD AND VARIANCE WITH CUTOFF ###############

timelag    = [n*dt for n in timelag]
theory_msd = [4*D*n + epsilon for n in timelag]
theory_variance = THEORETICAL_VARIANCE_2(N_array,sigma,D,dt)[:cutoff]

print("Compute the covariance matrix for the experimental MSD... This might take a few minutes...")

cov = COVARIANCE(data,cutoff)
theory_covariance = THEORETICAL_COVARIANCE_2(N_array,sigma,D,dt,cutoff)#[:cutoff,:cutoff]

print("Show experimental covariance for the MSD...")

print("Show theoretical covariance for the associated MSD...")


###########################################
############### ERROR SLOPE ###############
print("Compute the error on a linear fit of the MSD as a function of the number of fitting points, using the previously determined experimental variance and covariance...")

Pmin_array = [int(p) for p in np.linspace(2,cutoff-2,cutoff-3)]
sigmab_theory = []
sigmab_exp_th = []
sigmab_exp    = []

for p in Pmin_array:
	sigmab_theory.append(NormMSDSlopeError(N,sigma,D,dt,p))
	sigmab_exp_th.append(NormMSDSlopeError_exp(N,sigma,D,dt,theory_variance,theory_covariance,p))
	sigmab_exp.append(NormMSDSlopeError_exp(N,sigma,D,dt,var,cov,p))

print("Plot the relative error on the slope of the linear fit vs the number of fitting points...")

min_y = min(sigmab_exp) 
min_x = Pmin_array[sigmab_exp.index(min_y)]

print("It is estimated that the lowest error will be for a number of fitting points P = ",min_x," for which the relative error sigma/b = ",min_y)


timelag = np.array(timelag)
x = timelag.reshape((-1, 1))
y = msd
ctffb = int(min_x)
model = LinearRegression().fit(x[:ctffb], y[:ctffb])
r_sq = model.score(x[:ctffb], y[:ctffb])
fit_b = model.predict(x)

Dvalue = round(model.coef_[0]/4,4)
Dvalue_error = round(min_y*model.coef_[0]/4,4)


###########################################
############### ERROR INTERCEPT ###############
print("Compute the error on the intercept using the previously computed variance and covariance.")

Pmin_array = [int(p) for p in np.linspace(2,cutoff-2,cutoff-3)]
sigmaa_theory = []
sigmaa_exp_th = []
sigmaa_exp    = []
for p in Pmin_array:
	sigmaa_theory.append(NormMSDInterceptError(N,sigma,D,dt,p))
	sigmaa_exp_th.append(NormMSDInterceptError_exp(N,sigma,D,dt,theory_variance,theory_covariance,p))
	sigmaa_exp.append(NormMSDInterceptError_exp(N,sigma,D,dt,var,cov,p))

min_y = min(sigmaa_exp) 
min_x = Pmin_array[sigmaa_exp.index(min_y)]

print("It is estimated that the lowest error will be when the number of fitting points P = ",min_x," for which the relative error sigmaa/a = ",min_y)

timelag = np.array(timelag)
xa = timelag.reshape((-1, 1))
ya = msd
ctffa = int(min_x)
model = LinearRegression().fit(x[:ctffa], y[:ctffa])
r_sq = model.score(x[:ctffa], y[:ctffa])
fit_a = model.predict(x)

loc_sigma = round(np.sqrt(model.intercept_/4),4)
loc_sigma_error = round(min_y/2*model.intercept_,4)


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
msd_plot.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(msd,var)],[a + np.sqrt(b) for a,b in zip(msd,var)], color="#b3d1ff",alpha=0.5)
msd_plot.fill_between(timelag,[a - np.sqrt(b) for a,b in zip(theory_msd,theory_variance)],[a + np.sqrt(b) for a,b in zip(theory_msd,theory_variance)], color="#febab3",alpha=0.5) 
msd_plot.plot(timelag,msd,color=cexp, lw=2,label="Experimental MSD $\pm \sigma $")
msd_plot.plot(timelag,theory_msd,color=cth, lw=2,label="Theoretical MSD $\pm \sigma $")
msd_plot.set_ylabel(r"Mean square displacement ($\mu$m$^2/$s)",fontsize=10)
msd_plot.set_xlabel('Time lag (s)',fontsize=10)
msd_plot.legend(loc="upper left",fontsize=10)

########## HISTOGRAM ########################"

hist.spines["top"].set_visible(False)  
hist.spines["right"].set_visible(False)
hist.get_xaxis().tick_bottom()  
hist.get_yaxis().tick_left()
hist_array = [n*dt for n in N_array]
hist.hist(hist_array,color=cexp, bins=int(len(N_array)))
hist.axvline(cutoff*dt, 0, max(hist_array),color=cth)
hist.set_xlabel('Track duration (s)',fontsize=10)
hist.set_ylabel('Number of tracks',fontsize=10)

######## COVARIANCE #############################

im1 = cov_exp.pcolormesh(timelag,timelag,cov,cmap="Blues")
cov_exp.set_xlabel(r'$n \Delta t$')
cov_exp.set_ylabel(r'$m \Delta t$')
cbar = plt.colorbar(im1,ax=cov_exp)
#plt.title('Map of experimental covariance values')

im2 = cov_th.pcolormesh(timelag,timelag,theory_covariance,cmap="Reds")
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
D_plot.plot(timelag,msd,"-x",label="Exp.",color=cexp,ms=msize)
D_plot.plot(timelag,theory_msd,label="Th.",color=cth)
D_plot.legend(loc="upper left",fontsize=8)
D_plot.set_xlabel('Timelag (s)')
D_plot.set_ylabel(r'MSD $\mu$m$^2/$s')
D_plot.text(max(timelag)/5,max(msd)/10,r'$D = ($'+str(Dvalue)+"$\pm$"+str(Dvalue_error)+") $\mu m^2 / s$",color="purple",fontsize=8)

loc_unc_plot.spines["top"].set_visible(False)  
loc_unc_plot.spines["right"].set_visible(False)
loc_unc_plot.get_xaxis().tick_bottom()  
loc_unc_plot.get_yaxis().tick_left()
loc_unc_plot.plot(x, fit_a,"-x",label=r'$P = $'+str(ctffa),color="purple",ms=msize)
loc_unc_plot.plot(timelag,msd,"x-",label="Exp.",color=cexp,ms=msize)
loc_unc_plot.plot(timelag,theory_msd,label="Th.",color=cth)
loc_unc_plot.legend(loc="upper left",fontsize=8)
loc_unc_plot.set_xlabel('Timelag (s)')
loc_unc_plot.set_ylabel(r'MSD $\mu$m$^2/$s')
loc_unc_plot.text(max(timelag)/3,max(msd)/10,r'$\sigma_0 = ($'+str(loc_sigma)+"$\pm$"+str(loc_sigma_error)+")",color="purple",fontsize=8)

fig.suptitle(r'Determination of the diffusion coefficient for tracks in '+filename+" with initial D = "+str(D)+", $\sigma_0 = $"+str(sigma)+"\n for the cluster colored in "+clustercolor, fontsize=16)
plt.savefig(output_folder+'/graphs.png')
plt.show()
exit()
