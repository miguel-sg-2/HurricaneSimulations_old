#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: 
    
@author: Miguel Sanchez Gomez

Created on Wed Feb  8 08:40:11 2023

"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
################ ESTIMATE CENTER OF THE HURRICANE AT EACH #####################
################### HEIGHT BASED ON MINIMUM WIND SPEED ########################
###############################################################################

#%% Import modules
from IPython import get_ipython
get_ipython().magic('reset -sf')
import os
import numpy as np   
import matplotlib.pyplot as mpyplot
import matplotlib.dates as mdates 
import matplotlib.colors as mcolors 
import matplotlib.style as mstyle
import pandas as pandas
from netCDF4 import Dataset
import cmocean
import datetime
import scipy as scipy
import pandas
import xarray as xr              
from scipy import stats,signal
from matplotlib import ticker,colors

print("done importing modules")


#%% Define file locations
WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/newSetup/sst_34_NBA/"

WRF_FILES = 'merged_d05.nc' 

#%% Extract variables from file
ds=xr.open_dataset(WRF_DIRECTORY+WRF_FILES,decode_times=False)
# Wind speed
u = ds['U']
u = np.array(u[:,0,:,:,:])
v = ds['V']
v = np.array(v[:,0,:,:,:])
w = ds['W']
w = np.array(w[:,0,:,:,:])
# De-stagger wind speed
u_des = 0.5*(u[:,:,:,0:np.shape(u)[3]-1]+u[:,:,:,1:np.shape(u)[3]])
u = 0.0
v_des = 0.5*(v[:,:,0:np.shape(v)[2]-1,:]+v[:,:,1:np.shape(v)[2],:])
v = 0.0
w_des = 0.5*(w[:,0:np.shape(w)[1]-1,:,:]+w[:,1:np.shape(w)[1],:,:])
w = 0.0
# Calculate horizontal wind speed
uv = np.sqrt(np.array(u_des)**2 + np.array(v_des)**2)
# Perturbation Pressure
PP = ds['P']
PP = np.array(PP[:,0,:,:,:])
# Base state pressure
PB = ds['PB']
PB = np.array(PB[:,0,:,:,:])
P_tot = PP + PB
# Potential temperature
t = ds['T']
theta = np.array(t[:,:,:,:]) + 300
t_skin = ds['TSK']
t_skin = np.array(t_skin[:,0,:,:]) 
#t_skin = np.mean(np.mean(t_skin,axis=1),axis=1)
# Height variables
ph = ds['PH']
ph = np.array(ph[-1,0,:,:,:])
phb = ds['PHB']
phb = np.array(phb[-1,0,:,:,:])
height_stag = (ph[:,0,0]+phb[:,0,0])/9.81
height_agl = 0.5*(height_stag[1:] + height_stag[0:-1])
# Time
Time = ds['Times']
Time = np.array(Time)
# Domain dimensions
dx = ds.DX
n_x = dx*np.arange(0,np.shape(uv)[3])
n_y = dx*np.arange(0,np.shape(uv)[2])
ds.close()

# Turbine specifications
D = 126
z_hh = 90
rated_ws = 11.4

print("done extracting variables")

#%% Create time vector (after initialization) (ignore dates)
# Create large datetime64 array
ttime = np.arange('2000-01-01T00:00:00', '2000-01-02T00:00:00',np.shape(Time)[0], dtype='datetime64')
# Save actual times for simulation
for i_t in np.arange(0,np.shape(Time)[0]):
    a = str(Time[i_t])
    ttime[i_t] = np.datetime64(a[3:13] + str(' ')+ a[14:-2])
    # ttime[i_t] = ttime[0] + np.timedelta64(5*60,dtype='datetime64[s]')*i_t
# Save portion of temp array that actually matters
ttime = ttime[0:np.shape(Time)[0]]

print(ttime)

#%% Time since initialization
t_init = np.datetime64('2000-01-01T00:00:18')

time_sinceInit = ttime - t_init
time_sinceInit = time_sinceInit.astype('timedelta64[s]')

t_since = time_sinceInit.astype('float')/3600 # [hr]

#%% Find hub-height
i_hh = np.argmin(np.abs(height_agl - z_hh))

#%% Plot wind speed at hub-height
iz = 0 # np.argmin(np.abs(height_agl - z_hh))
it = -1

# Horizontal wind speed
mpyplot.figure(figsize=(6,4))
mpyplot.title('t = '+str(ttime[it])[8:10]+' - '+str(ttime[it])[11:])
im1 = mpyplot.pcolormesh(n_x/1000,n_y/1000,uv[it,iz,:,:])
cbar = mpyplot.colorbar(im1)
cbar.set_label('U @ ' + str(int(height_agl[iz])) + 'm AGL [m s$^{-1}$]',fontsize=12)
mpyplot.ylabel('y [km]',fontsize=14)
mpyplot.xlabel('x [km]',fontsize=14)
#mpyplot.scatter(n_x[4]/1000,n_y[4]/1000,color='r')
#if 'merged_d05.nc' in WRF_FILES:
#    mpyplot.scatter(n_x[locs_ts[:,0]]/1000,n_y[locs_ts[:,1]]/1000,s=5,color='red')
#mpyplot.xlim(143*dx/1000,(143*dx + 403*4500)/1000)
#mpyplot.ylim(143*dx/1000,(143*dx + 403*4500)/1000)
#mpyplot.ylim(2000,4000)
#mpyplot.xlim(2000,4000)
mpyplot.show()
mpyplot.close()


# Zonal wind speed
mpyplot.figure(figsize=(6,4))
u_des[it,iz,0,0] = 0.5
u_des[it,iz,0,1] = -0.5
newCmap = cmocean.tools.crop(mpyplot.cm.RdBu, np.min(u_des[it,iz,:,:]), np.max(u_des[it,iz,:,:]), 0)
mpyplot.title('t = '+str(ttime[it])[8:10]+' - '+str(ttime[it])[11:])
im1 = mpyplot.pcolormesh(n_x/1000,n_y/1000,u_des[it,iz,:,:],cmap=newCmap)
cbar = mpyplot.colorbar(im1)
cbar.set_label('u @ ' + str(int(height_agl[iz])) + 'm AGL [m s$^{-1}$]',fontsize=12)
mpyplot.ylabel('y [km]',fontsize=14)
mpyplot.xlabel('x [km]',fontsize=14)
#mpyplot.ylim(2000,4000)
#mpyplot.xlim(2000,4000)
mpyplot.show()
mpyplot.close()

# Meridional wind speed
mpyplot.figure(figsize=(6,4))
v_des[it,iz,0,0] = 0.5
v_des[it,iz,0,1] = -0.5
newCmap = cmocean.tools.crop(mpyplot.cm.RdBu, np.min(v_des[it,iz,:,:]), np.max(v_des[it,iz,:,:]), 0)
mpyplot.title('t = '+str(ttime[it])[8:10]+' - '+str(ttime[it])[11:])
im1 = mpyplot.pcolormesh(n_x/1000,n_y/1000,v_des[it,iz,:,:],cmap=newCmap)
cbar = mpyplot.colorbar(im1)
cbar.set_label('v @ ' + str(int(height_agl[iz])) + 'm AGL [m s$^{-1}$]',fontsize=12)
mpyplot.ylabel('y [km]',fontsize=14)
mpyplot.xlabel('x [km]',fontsize=14)
#mpyplot.ylim(400000/1000,600000/1000)
#mpyplot.xlim(400000/1000,600000/1000)
mpyplot.show()
mpyplot.close()


# Vertical wind speed
mpyplot.figure(figsize=(6,4))
newCmap = cmocean.tools.crop(mpyplot.cm.RdBu, np.min(w_des[it,iz,:,:]), np.max(w_des[it,iz,:,:]), 0)
im1 = mpyplot.pcolormesh(n_x/1000,n_y/1000,w_des[it,iz,:,:],cmap=newCmap)
cbar = mpyplot.colorbar(im1)
cbar.set_label('w @ ' + str(int(height_agl[iz])) + 'm AGL [m s$^{-1}$]',fontsize=12)
mpyplot.ylabel('y [km]',fontsize=14)
mpyplot.xlabel('x [km]',fontsize=14)
#mpyplot.ylim(400000,600000)
#mpyplot.xlim(400000,600000)
mpyplot.show()
mpyplot.close()

#%% Minimum surface pressure
min_pp_s = np.min(np.reshape(P_tot[:,0,:,:],[len(ttime),len(n_x)*len(n_y)]),axis=1)

mpyplot.figure(figsize=(10,3))
mpyplot.plot(t_since[1:],min_pp_s[1:]/100)
mpyplot.xlabel('Time since initialization [hr]',fontsize=14)
mpyplot.ylabel(r'min P$_{z=10m}$ [hPa]',fontsize=14)
mpyplot.xlim(t_since[0],t_since[-1])
mpyplot.show()
mpyplot.close()

#%% Find center of hurricane for each height
center = np.zeros([len(ttime),len(height_agl),2]) + np.nan # [time,height,[x,y]]

X_grid,Y_grid = np.meshgrid(n_x,n_y)
X_grid = X_grid.flatten()
Y_grid = Y_grid.flatten()

for iz in np.arange(len(height_agl)):
    for iitt in np.arange(np.shape(uv)[0]):
        
        temp_uv = np.zeros(np.shape(uv[iitt,iz,:,:])) + uv[iitt,iz,:,:]
        temp_uv[0:10,:] = np.nan
        temp_uv[-10:,:] = np.nan
        temp_uv[:,0:10] = np.nan
        temp_uv[:,-10:] = np.nan
        uv_flat = temp_uv.flatten() #uv[-1,0,:,:].flatten()
        uv_flat[np.isnan(uv_flat)] = 0
        max_ws_ref = np.argmax(uv_flat)
        
        lim_low_x = 0.3*np.max(n_x)
        lim_low_y = 0.3*np.max(n_y)
        lim_high_x = 0.6*np.max(n_x)
        lim_high_y = 0.6*np.max(n_x)
            
        # Create small meshgrid
        small_x,small_y = np.meshgrid(n_x[(n_x<lim_high_x)&(n_x>lim_low_x)],n_y[(n_y<lim_high_y)&(n_y>lim_low_y)])
        flat_smallX = small_x.flatten()
        flat_smallY = small_y.flatten()
    
        # Crop array
        temp_uv_1 = uv[iitt,iz,(n_y<lim_high_y)&(n_y>lim_low_y),:]
        temp_uv_2 = temp_uv_1[:,(n_x<lim_high_x)&(n_x>lim_low_x)]
        # Flatten array
        temp_uv_flat = temp_uv_2.flatten()
        # Find location of minimum
        if np.any(temp_uv_flat)==True:
            i_min = np.argmin(temp_uv_flat)
            # Save location of minimum
            center[iitt,iz,0] = flat_smallX[i_min]
            center[iitt,iz,1] = flat_smallY[i_min]
        else:
            # Save location of minimum
            center[iitt,iz,0] = np.mean(n_x)
            center[iitt,iz,1] = np.mean(n_y)
    
#    # Show area considered for center of hurricane
#    mpyplot.figure()
#    mpyplot.pcolormesh(n_x/1000,n_y/1000,uv[iitt,iz,:,:]) 
#    mpyplot.plot(center[:,0]/1000,center[:,1]/1000,'o-',color='red',markersize=3)
#    mpyplot.fill(np.array([lim_low_x,lim_high_x,lim_high_x,lim_low_x])/1000,np.array([lim_low_y,lim_low_y,lim_high_y,lim_high_y])/1000,facecolor='none',edgecolor='r',linestyle=':',label='Area')
#    mpyplot.xlabel('x [km]')
#    mpyplot.ylabel('y [km]')
#    mpyplot.xlim(1800,2200)
#    mpyplot.ylim(1800,2200)
#    mpyplot.show()
#    mpyplot.close()   


    
# Show area considered for center of hurricane and hurricane track
mpyplot.figure()
mpyplot.pcolormesh(n_x/1000,n_y/1000,uv[-1,iz,:,:]) 
mpyplot.plot(center[:,iz,0]/1000,center[:,iz,1]/1000,'o-',color='red',markersize=3)
mpyplot.fill(np.array([lim_low_x,lim_high_x,lim_high_x,lim_low_x])/1000,np.array([lim_low_y,lim_low_y,lim_high_y,lim_high_y])/1000,facecolor='none',edgecolor='r',linestyle=':',label='Area')
mpyplot.xlabel('x [km]')
mpyplot.ylabel('y [km]')
#mpyplot.xlim((min(center[:,0]) - 50*dx)/1000,(max(center[:,0]) + 50*dx)/1000)
#mpyplot.ylim((min(center[:,1]) - 50*dx)/1000,(max(center[:,1]) + 50*dx)/1000)
mpyplot.show()
mpyplot.close()  


#%% Get geometrical center using velocity contours
slow_winds = [5,7,9]
geo_center2 = np.zeros([len(ttime),len(slow_winds),2])

dyn_center = center
for iitt in np.arange(len(ttime)):
    ## Limits on what to plot
    lim_low_x = dyn_center[iitt,0,0] - 0.5*np.mean(n_x)
    lim_low_y = dyn_center[iitt,0,1] - 0.5*np.mean(n_y)
    lim_high_x = dyn_center[iitt,0,0] + 0.5*np.mean(n_x)
    lim_high_y = dyn_center[iitt,0,1] + 0.5*np.mean(n_y)
    
    
    # Get contours for a set of slow wind speeds close to the hurricane center
    small_x = n_x[(n_x>lim_low_x) & (n_x<lim_high_x)]
    small_y = n_y[(n_y>lim_low_y) & (n_y<lim_high_y)]
    small_uv = uv[iitt,iz,(n_y>lim_low_y) & (n_y<lim_high_y),:]
    small_uv = small_uv[:,(n_x>lim_low_x) & (n_x<lim_high_x)]
    cs = mpyplot.contour(small_x/1000,small_y/1000,small_uv, slow_winds)
    paths = cs.collections
    mpyplot.close()
    
#    mpyplot.figure()
#    mpyplot.contourf(small_x/1000,small_y/1000,small_uv)
#    mpyplot.contour(small_x/1000,small_y/1000,small_uv, slow_winds,cmap=mpyplot.cm.binary)
#    mpyplot.show()
    
    # Get vertices of contours for each wind speed
    for i_ws in np.arange(len(slow_winds)):
        allP = paths[i_ws].get_paths()
        # Find longest path
        longest = 0
        len_longest = 0
        if len(allP)>0:
            for i_p in np.arange(len(allP)):
                if len(allP[i_p]) > len_longest:
                    longest = i_p
                    len_longest = len(allP[i_p])
            path = allP[longest]
            verts = path.vertices
            x_contour = verts[:,0]
            y_contour = verts[:,1]
            geo_center2[iitt,i_ws,0] = np.mean(x_contour)*1000
            geo_center2[iitt,i_ws,1] = np.mean(y_contour)*1000
        
#%% Spatial mean using hurricane's geometrical center as reference
iz = 0
# Maximum distance from radius
max_dist_fromCent = 35000

# Initialize array
uv_r_center = np.zeros([len(ttime),int(2*max_dist_fromCent/dx),int(2*max_dist_fromCent/dx)]) + np.nan

for iitt in np.arange(1,len(ttime)):
    # Geometrical center
    geo_cent = [geo_center2[iitt,2,0],geo_center2[iitt,2,1]]
#    geo_cent = [center[iitt,0,0],center[iitt,0,1]]
    ix_c = int(geo_cent[0]/dx)
    iy_c = int(geo_cent[1]/dx)
    # Save portion of hurricane
    uv_r_center[iitt,:,:] = uv[iitt,iz,iy_c-int(max_dist_fromCent/dx):iy_c+int(max_dist_fromCent/dx)+1,ix_c-int(max_dist_fromCent/dx):ix_c+int(max_dist_fromCent/dx)+1]
    
# Temporal average
mean_uv_r = np.nanmean(uv_r_center,axis=0)
n_xx = np.arange(np.shape(mean_uv_r)[1])*dx
n_yy = np.arange(np.shape(mean_uv_r)[0])*dx
n_xx = n_xx - np.mean(n_xx)
n_yy = n_yy - np.mean(n_yy)

# Figure
mpyplot.figure()
#im1 = mpyplot.contourf(n_xx/1000,n_yy/1000,mean_uv_r,levels=10) 
im1 = mpyplot.pcolormesh(n_xx/1000,n_yy/1000,mean_uv_r) 
mpyplot.xlabel('x [km]',fontsize=14)
mpyplot.ylabel('y [km]',fontsize=14)
cbar = mpyplot.colorbar(im1)
cbar.set_label(r'$\overline{U}$ @ ' + str(int(height_agl[iz])) + 'm AGL [m s$^{-1}$]',fontsize=12)
mpyplot.savefig(WRF_DIRECTORY + "time_avg_U.png",facecolor='w',edgecolor='w',dpi=400,bbox_inches='tight')
mpyplot.show()
mpyplot.close()  


mpyplot.figure()
mpyplot.plot(n_xx/1000,mean_uv_r[int(0.5*len(n_yy)),:])
mpyplot.plot(n_yy/1000,mean_uv_r[:,int(0.5*len(n_xx))])
mpyplot.xlim(-30,30)
mpyplot.xlabel('r [km]',fontsize=14)
mpyplot.ylabel('$\overline{U}$ [m s$^{-1}$',fontsize=14)
mpyplot.savefig(WRF_DIRECTORY + "r_hurr.png",facecolor='w',edgecolor='w',dpi=400,bbox_inches='tight')

#%% Save array with center location of hurricane for each height
# x-location of hurricane center
header_ = 'time,x location for center at z_i'
with open(WRF_DIRECTORY + 'center_x.txt', 'w') as f:
    for i in np.arange(len(ttime)+1):
        if i == 0:
            f.write(header_)
            f.write('\n')
        else:
            arr = center[i-1,:,0]
            line = str(time_sinceInit[i-1].astype('double')) + ','
            for j in np.arange(len(arr)):
                line = line+str(arr[j])
                if j<len(arr)-1:
                    line = line+','
            f.write(line)
            f.write('\n')
                
# y-location of hurricane center
header_ = 'time,y location for center at z_i'
with open(WRF_DIRECTORY + 'center_y.txt', 'w') as f:
    for i in np.arange(len(ttime)+1):
        if i == 0:
            f.write(header_)
            f.write('\n')
        else:
            arr = center[i-1,:,1]
            line = str(time_sinceInit[i-1].astype('double')) + ','
            for j in np.arange(len(arr)):
                line = line+str(arr[j])
                if j<len(arr)-1:
                    line = line+','
            f.write(line)
            f.write('\n')
