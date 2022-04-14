#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:54:29 2022

@author: noam
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
import os
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
from datetime import timedelta, datetime
from tqdm import tqdm
from glob import glob

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
# Using parameters for A. millepora from Connolly et al. (2010)
parameters = {'t0' : datetime(year=2019, month=10, day=1, hour=0),
              'dt' : timedelta(minutes=30),
              'xlim' : [38, 50],
              'ylim' : [-13, -6],
              'min_dens' : 8e1,
              'n_frames' : 120,
              'fps' : 60,
              'sf'  : 8000}

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
fh['traj_data'] = np.array(sorted(glob(dirs['traj'] + 'WINDS_Aldabra_release_*.nc')))
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
# GET A LIST OF TIME OFFSETS FOR INPUT FILES                                 #
##############################################################################

t0_list = []

for fhi in fh['traj_data']:
    t0_list.append(datetime(year=int(fhi.split('_')[-5]),
                            month=int(fhi.split('_')[-4]),
                            day=int(fhi.split('_')[-3]),
                            hour=0))

t0_list = np.array(t0_list)
t_offset = ((t0_list - t0_list.min())/parameters['dt']).astype(int)
fh0 = fh['traj_data'][np.where(t_offset == 0)[0]][0]

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

def return_l2mean(plon, plat, lon_bnds, lat_bnds, l2):
    weighted = np.histogram2d(plon, plat, bins=[lon_bnds, lat_bnds],
                              weights=l2)[0].T
    distro = np.histogram2d(plon, plat, bins=[lon_bnds, lat_bnds])[0].T
    distro[distro == 0] = 1
    histo = weighted/distro

    return np.ma.masked_where(histo == 0, histo)

def return_l2(plon, plat, lon_bnds, lat_bnds, l2):
    weighted = np.histogram2d(plon, plat, bins=[lon_bnds, lat_bnds],
                              weights=l2)[0].T

    return np.ma.masked_where(weighted == 0, weighted)

with Dataset(fh0, mode='r') as nc:
    p_lon = nc.variables['lon'][:, 4]
    p_lat = nc.variables['lat'][:, 4]
    p_l2 = nc.variables['L2'][:, 4]*parameters['sf']
    p_l2[p_l2 == 0] = parameters['min_dens']

hist = ax.pcolormesh(lon_bnd, lat_bnd, return_l2(p_lon, p_lat, lon_bnd, lat_bnd, p_l2),
                     cmap=cmr.ember, norm=colors.LogNorm(vmin=1e2, vmax=1e7), zorder=3)

div = make_axes_locatable(ax)
cax = div.append_axes('right', size='2%', pad=0.10)
cbar = f.colorbar(hist, cax=cax, orientation='vertical')
cbar.set_label('Number of competent larvae per grid cell', fontsize=16)

##############################################################################
# ANIMATE                                                                    #
##############################################################################

pbar = tqdm(total=parameters['n_frames']+1)

def update_plot(frame):
    # Determine which files are valid
    valid_fh_list = fh['traj_data'][(t_offset <= frame)*(t_offset+5760 >= frame)]
    valid_fh_offset_list = t_offset[(t_offset <= frame)*(t_offset+5760 >= frame)]

    global p_lon
    global p_lat
    global p_l2
    p_lon, p_lat, p_l2 = np.zeros((1,)), np.zeros((1,)), np.zeros((1,))

    for fhi, offset in zip(valid_fh_list, valid_fh_offset_list):
        with Dataset(fhi, mode='r'):
            if fhi == valid_fh_list[0]:
                p_lon = nc.variables['lon'][:, frame-offset]
                p_lat = nc.variables['lat'][:, frame-offset]
                p_l2 = nc.variables['L2'][:, frame-offset]*parameters['sf']
            else:
                p_lon = np.concatenate([p_lon, nc.variables['lon'][:, frame-offset]])
                p_lat = np.concatenate([p_lat, nc.variables['lat'][:, frame-offset]])
                p_l2 = np.concatenate([p_l2, nc.variables['L2'][:, frame-offset]])*parameters['sf']

    p_l2[p_l2 == 0] = parameters['min_dens']
    hist.set_array(return_l2(p_lon, p_lat, lon_bnd, lat_bnd, p_l2).ravel())
    time_str.set_text((frame_time+frame*parameters['dt']).strftime('%d/%m/%Y, %H:%M'))

    pbar.update(1)

    return hist, time_str

ani = animation.FuncAnimation(f, update_plot, parameters['n_frames'],
                              interval=1000/parameters['fps'])
ani.save(fh['out'], fps=parameters['fps'], bitrate=16000,)







