#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:54:29 2022

@author: noam
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import cmasher as cmr
import os
import matplotlib.animation as animation
from netCDF4 import Dataset
from datetime import timedelta, datetime
from tqdm import tqdm

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
# Using parameters for A. millepora from Connolly et al. (2010)
parameters = {'t0' : datetime(year=2019, month=10, day=1, hour=0),
              'dt' : timedelta(minutes=30),
              'xlim' : [38, 50],
              'ylim' : [-13, -6],
              'min_dens' : 1e-9,
              'n_frames' : 480,
              'fps' : 48}

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/WINDS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/Aldabra/'

# FILEHANDLES
fh = {}
fh['winds_grid'] = dirs['grid'] + 'griddata_winds.nc'
fh['coral_grid'] = dirs['grid'] + 'coral_grid.nc'
fh['traj_data'] = dirs['traj'] + 'WINDS_Aldabra_release_2019_10_1_1_1.nc'
fh['out'] = dirs['fig'] + 'Aldabra_test_animation.mp4'

##############################################################################
# LOAD GRID DATA                                                             #
##############################################################################

with Dataset(fh['winds_grid'], mode='r') as nc:
    lon = nc.variables['lon_rho'][0, :]
    lat = nc.variables['lat_rho'][:, 0]

    lon_bnd = nc.variables['lon_u'][0, :]
    lat_bnd = nc.variables['lat_v'][:, 0]
    lon_bnd = np.roll(np.append(lon_bnd, [2*lon_bnd[-1]-lon_bnd[-2],
                                2*lon_bnd[0]-lon_bnd[1]]), 1)
    lat_bnd = np.roll(np.append(lat_bnd, [2*lat_bnd[-1]-lat_bnd[-2],
                                2*lat_bnd[0]-lat_bnd[1]]), 1)
    h = nc.variables['h'][:]

with Dataset(fh['coral_grid'], mode='r') as nc:
    rc = nc.variables['reef_cover_w'][:]
    lsm = nc.variables['lsm_w'][:]

##############################################################################
# SET UP BASE SCENE                                                          #
##############################################################################

f, ax = plt.subplots(1, 1, figsize=(20,10))
ax.set_aspect('equal')

bath = ax.pcolormesh(lon_bnd, lat_bnd, h, vmin=0, vmax=10000,
                     cmap=cmr.ocean_r, zorder=1)
land = ax.pcolormesh(lon_bnd, lat_bnd, np.ma.masked_where(lsm == 0, lsm),
                     vmin=0, vmax=3, cmap=cmr.neutral, zorder=2)
coral = ax.pcolormesh(lon_bnd, lat_bnd, np.ma.masked_where(rc == 0, rc),
                      cmap=cmr.swamp_r, norm=colors.LogNorm(vmin=1e4, vmax=1e1*rc.max()),
                      zorder=3, alpha=1)

x_ticks = np.arange(35, 80, 2)
y_ticks = np.arange(-25, 5, 2)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

for i in x_ticks:
    ax.plot([i, i], [-25, 5], linewidth=0.5, linestyle='--', color='k')

for j in y_ticks:
    ax.plot([30, 80], [j, j], linewidth=0.5, linestyle='--', color='k')

if parameters['xlim'] == None:
    ax.set_xlim([lon_bnd[0], lon_bnd[-1]])
else:
    ax.set_xlim(parameters['xlim'])

if parameters['ylim'] == None:
    ax.set_ylim([-23.5, 0])
else:
    ax.set_ylim(parameters['ylim'])

ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

frame_time = parameters['t0']

time_str = ax.text(ax.get_xlim()[-1]-0.1, ax.get_ylim()[-1]-0.1,
                   frame_time.strftime('%d/%m/%Y, %H:%M'), fontsize=16,
                   horizontalalignment='right', verticalalignment='top',
                   fontfamily='ubuntu', fontweight='bold', color='w')

with Dataset(fh['traj_data'], mode='r') as nc:
    p_lon = nc.variables['lon'][:]
    p_lat = nc.variables['lat'][:]
    n_traj = len(p_lat)

    p_l2 = nc.variables['L2'][:]/(n_traj*50)
    p_l2[p_l2 == 0] = parameters['min_dens']

p_density = np.histogram2d(p_lon[:, 0], p_lat[:, 0], bins=[lon_bnd, lat_bnd],
                           weights=p_l2[:, 0])[0].T

hist = ax.pcolormesh(lon_bnd, lat_bnd, np.ma.masked_where(p_density == 0, p_density),
                     cmap=cmr.ember, norm=colors.LogNorm(vmin=parameters['min_dens'], vmax=1e-1),
                     zorder=3)


##############################################################################
# ANIMATE                                                                    #
##############################################################################

pbar = tqdm(total=parameters['n_frames']+1)

def update_plot(frame):
    p_density = np.histogram2d(p_lon[:, frame], p_lat[:, frame], bins=[lon_bnd, lat_bnd],
                               weights=p_l2[:, frame])[0].T
    hist.set_array(p_density.ravel())

    time_str.set_text((frame_time+frame*parameters['dt']).strftime('%d/%m/%Y, %H:%M'))

    pbar.update(1)


ani = animation.FuncAnimation(f, update_plot, parameters['n_frames'],
                              interval=1000/parameters['fps'])

ani.save(fh['out'], fps=parameters['fps'], bitrate=16000,)







