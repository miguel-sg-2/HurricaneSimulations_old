#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: 
    
@author: Miguel Sanchez Gomez

Created on Fri Apr 22 17:36:32 2022

"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
######## RE-WRITE WRFINPUT FILE OF DOMAIN USING VELOCITY FIELDS ###############
##################### FROM INCIPIENT TROPICAL CYCLONE #########################
###############################################################################

#%% Import modules
from IPython import get_ipython
get_ipython().magic('reset -sf')
import os
import numpy as numpy   
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

#%% Define file location
#WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/sst_26/"
#WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/newSetup/sst_26/"
#WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/newSetup/modified_inputSounding/sst_26/"
#WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/newSetup/sst_28/"
#WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/newSetup/sst_30/"
WRF_DIRECTORY = "/Users/misa5952/Documents/PhD/Research/Hurricanes_and_turbines/Files/newSetup/sst_34/"

input_largeDomain = "wrfrst_d01_2000-01-01_00:00:30" # "wrfrst_d01_2000-01-01_00:00:10"
modified_input = "wrfrst_d01_2000-01-01_00:00:30_mod" #"wrfrst_d01_2000-01-01_00:00:10_mod"


#%% Extract variables from input of large domain
ds=xr.open_dataset(WRF_DIRECTORY+input_largeDomain,decode_times=False)
# Wind speed
u = ds['U_1']
u_np = numpy.array(u[:,:,:,:])
v = ds['V_1']
v_np = numpy.array(v[:,:,:,:])
# Sea surface temperautre
sst = ds['TSK']
# Height variables
ph = ds['PH_1']
ph = numpy.array(ph[0,:,:,:])
phb = ds['PHB']
phb = numpy.array(phb[0,:,:,:])
height_stag = (ph+phb)/9.81
ph = 0.0
phb = 0.0
height_agl = (height_stag[1:,:,:]+height_stag[0:-1,:,:])/2
height_agl = numpy.array(numpy.mean(numpy.mean(height_agl,axis=1),axis=1))
height_stag = numpy.mean(numpy.mean(height_stag,axis=1),axis=1)
# Domain dimensions
dx = ds.DX


print("done extracting variables for input, large domain simulation")
ds.close()

#%% Parameters for hurricane
# Height of zero wind speed
z_0 = 20000 # m
# Radius of zero wind speed
r_0 = 412500 # m
# Maximum wind speed
v_max = 15 # m/s
#v_max = 50 # m/s

# Radius of maximum wind speed
r_max = 82500 # m
# Coriolis parameter for f-plane
Omega = 7.2921*(10**-5) # rad/s
f = 2*Omega*numpy.sin(20*numpy.pi/180) # 1/s


# Sea surface temperature
SST_hurr = 34 + 273.15
sst[:,:,:] = SST_hurr

print('SST is %f deg C' % (SST_hurr - 273.15))

#%% Create grid in cartesian coordinates
ddxx = dx
ddzz = 5
x = numpy.arange(-1*r_0,r_0,ddxx)
y = numpy.arange(-1*r_0,r_0,ddxx)
z = numpy.arange(0,z_0,ddzz)

#%% Tangential velocity in vortex using cartensian coordinates
# Initialize array
v_t_xy = numpy.zeros([len(z),len(y),len(x)])

# Calculate tangential veloctity at each location
for ix in numpy.arange(0,len(x)):
    for iy in numpy.arange(0,len(y)):
        for iz in numpy.arange(0,len(z)):
            # Radius at given (x,y) location
            rr = numpy.sqrt(x[ix]**2 + y[iy]**2)
            
            # Tangential velocity inside of max radius 
            if rr<=r_0:
                # Intensity decays with height
                I = (z_0-z[iz])/z_0
                # Term 1
                term1 = (v_max**2)*((rr/r_max)**2)
                # Term 2
                term2 = ( (2*r_max/(rr+r_max))**3 ) - ( (2*r_max/(r_0+r_max))**3 )
                # Term 3
                term3 = (f**2)*(rr**2)/4
                # Term 4
                term4 = f*rr/2
                
                # Tangential velocity
                v_t_xy[iz,iy,ix] = I*((term1*term2 + term3)**0.5 - term4 )

#%% Plot of tangential velocity
# X-Z slice
mpyplot.figure(figsize=(7,3))
mpyplot.contourf(x/1000,z/1000,v_t_xy[:,int(0.5*len(y)),:])
mpyplot.xlabel('x-distance [km]',fontsize=12)
mpyplot.ylabel('Height [km]',fontsize=12)
cbar = mpyplot.colorbar()
cbar.set_label(r'V$_t$ [m s$^{-1}$]',fontsize=14)
mpyplot.show()
mpyplot.close()

# X-Y Planview
mpyplot.figure(figsize=(7,6))
mpyplot.contourf(x/1000,y/1000,v_t_xy[0,:,:])
mpyplot.xlabel('x-distance [km]',fontsize=12)
mpyplot.ylabel('y-distance [km]',fontsize=12)
cbar = mpyplot.colorbar()
cbar.set_label(r'V$_t$ [m s$^{-1}$]',fontsize=14)
mpyplot.show()
mpyplot.close()
            
#%% Calcualate zonal and meridional components of wind speed at each location
# Magnitude of tangential velocity
v_t_magnitude = numpy.abs(v_t_xy)

# Initialize arrays
u_hurr = numpy.zeros(numpy.shape(v_t_xy))
v_hurr = numpy.zeros(numpy.shape(v_t_xy))

# Calculate zonal and meridional veloctity components at each location
for ix in numpy.arange(0,len(x)):
    for iy in numpy.arange(0,len(y)):
        # Hurricanes are cyclonic systems
        
        # Determine quadrant in cartesian coordinates
        if (x[ix]>0) & (y[iy]>0):
            # Angle to x-axis
            thet = numpy.arctan(y[iy]/x[ix])
            # Zonal velocity component
            u_hurr[:,iy,ix] = -1*v_t_magnitude[:,iy,ix]*numpy.sin(thet)
            # Meridional velocity component
            v_hurr[:,iy,ix]= v_t_magnitude[:,iy,ix]*numpy.cos(thet)
        elif (x[ix]<0) & (y[iy]>0):
            # Angle to x-axis
            thet = numpy.arctan(numpy.abs(y[iy]/x[ix]))
            # Zonal velocity component
            u_hurr[:,iy,ix] = -1*v_t_magnitude[:,iy,ix]*numpy.sin(thet)
            # Meridional velocity component
            v_hurr[:,iy,ix] = -1*v_t_magnitude[:,iy,ix]*numpy.cos(thet)
        elif (x[ix]<0) & (y[iy]<0):
            # Angle to y-axis
            thet = numpy.arctan(numpy.abs(x[ix]/y[iy]))
            # Zonal velocity component
            u_hurr[:,iy,ix] = v_t_magnitude[:,iy,ix]*numpy.cos(thet)
            # Meridional velocity component
            v_hurr[:,iy,ix] = -1*v_t_magnitude[:,iy,ix]*numpy.sin(thet)
        elif (x[ix]>0) & (y[iy]<0):
            # Angle to y-axis
            thet = numpy.arctan(numpy.abs(y[iy]/x[ix]))
            # Zonal velocity component
            u_hurr[:,iy,ix] = v_t_magnitude[:,iy,ix]*numpy.sin(thet)
            # Meridional velocity component
            v_hurr[:,iy,ix] = v_t_magnitude[:,iy,ix]*numpy.cos(thet)
        
# Plot velocity field
# X-Z slice
mpyplot.figure(figsize=(7,3))
mpyplot.contourf(x/1000,z/1000,v_hurr[:,int(0.5*len(y)),:],cmap = mpyplot.cm.RdBu)
mpyplot.xlabel('x-distance [km]',fontsize=12)
mpyplot.ylabel('Height [km]',fontsize=12)
cbar = mpyplot.colorbar()
cbar.set_label(r'v [m s$^{-1}$]',fontsize=14)
mpyplot.show()
mpyplot.close()

#%% X-Y Planview
mpyplot.figure(figsize=(8,4))
mpyplot.subplot(1,2,1)
mpyplot.title('u-velocity component')
mpyplot.contourf(x/1000,y/1000,u_hurr[0,:,:],cmap = mpyplot.cm.RdBu)
mpyplot.xlabel('x-distance [km]',fontsize=12)
mpyplot.ylabel('y-distance [km]',fontsize=12)
mpyplot.subplot(1,2,2)
mpyplot.title('v-velocity component')
mpyplot.contourf(x/1000,y/1000,v_hurr[0,:,:],cmap = mpyplot.cm.RdBu)
mpyplot.xlabel('x-distance [km]',fontsize=12)
cbar = mpyplot.colorbar()
cbar.set_label(r'u,v [m s$^{-1}$]',fontsize=14)
mpyplot.show()
mpyplot.close()


#%% Interpolate from hurricane grid to WRF grid
# Find center of domain
ix_mid = int(0.5*(numpy.shape(u)[3]-1))
iy_mid = int(0.5*(numpy.shape(v)[2]-1))
L_yy = int(0.5*numpy.shape(u_hurr)[1])
L_xx = int(0.5*numpy.shape(u_hurr)[2])

# RE-WRITE velocity field for each vertical level
for iz in numpy.arange(len(height_agl)):
    # Find closest vertical level
    i_close = numpy.argmin(numpy.abs(z - height_agl[iz]))
    
    # Re-write zonal velocity field
    u[0,iz,iy_mid-L_yy:iy_mid+L_yy,ix_mid-L_xx:ix_mid+L_xx] = u_hurr[i_close,:,:]
    
    # Re-write meridional velocity field
    v[0,iz,iy_mid-L_yy:iy_mid+L_yy,ix_mid-L_xx:ix_mid+L_xx] = v_hurr[i_close,:,:]

#%% Rewrite wrfinput file using extrapolated, turbulent fields
ds=xr.open_dataset(WRF_DIRECTORY+input_largeDomain)
# Re-write zonal winds
#ds['U']=xr.where((ds['U']>-10000),u_stag_LG,ds['U'])
ds = ds.drop_vars('U_1')
ds = ds.assign(U_1=u)
ds = ds.drop_vars('U_2')
ds = ds.assign(U_2=u)
# Re-write meridional winds
#ds['V']=xr.where((ds['V']<10000),v_stag_LG,v_stag_extrap)
ds = ds.drop_vars('V_1')
ds = ds.assign(V_1=v)
ds = ds.drop_vars('V_2')
ds = ds.assign(V_2=v)
# Re-write sea surface temperature
ds = ds.drop_vars('TSK')
ds = ds.assign(TSK=sst)
# Re-write file
ds.to_netcdf(WRF_DIRECTORY+modified_input)
ds.close()

#%% Verify the fields are modified
# Extract variables from initial input file
ds=xr.open_dataset(WRF_DIRECTORY+input_largeDomain)
u_init = ds['U_1']
v_init = ds['V_1']
ds.close()
ds = 0

# Extract variables from modified input file
ds=xr.open_dataset(WRF_DIRECTORY+modified_input)
u_mod = ds['U_1']
v_mod = ds['V_1']
ds.close()
ds = 0

# Figures
mpyplot.figure(figsize=(6,6))
# Initial files
mpyplot.subplot(2,2,1)
mpyplot.title('Initial fields',x=0.5,y=1.01)
mpyplot.pcolormesh(u_init[0,13,:,:],cmap = mpyplot.cm.RdBu)
mpyplot.subplot(2,2,3)
mpyplot.pcolormesh(v_init[0,13,:,:],cmap = mpyplot.cm.RdBu)
# Modified files
mpyplot.subplot(2,2,2)
mpyplot.title('Modified fields',x=0.5,y=1.01)
mpyplot.pcolormesh(u_mod[0,13,:,:],cmap = mpyplot.cm.RdBu)
mpyplot.subplot(2,2,4)
mpyplot.pcolormesh(v_mod[0,13,:,:],cmap = mpyplot.cm.RdBu)

mpyplot.tight_layout()
mpyplot.show()
mpyplot.close()