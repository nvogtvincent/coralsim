#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from CMEMS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from sys import argv
from coralsim import Experiment

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
mortality_t = 14         # Mortality timescale (d)
settling_t = 1           # Settling timescale (d)
mean_competency_t = 7    # Time at which 50% of larvae are competent (d)
competency_t_spread = 1  # Spread of logistic function for competency (d)

parameters = {'lm': 1/(mortality_t*24*3600),
              'ls': 1/(settling_t*24*3600),
              'tc': mean_competency_t*24*3600,
              'kc': 1/(competency_t_spread*24*3600)}

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/CMEMS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/'

# FILE HANDLES
# fh = {}
# fh['traj'] = sorted(glob(dirs['traj'] + 'cmems6400*.nc'))
# fh['grid'] = dirs['grid'] + 'coral_grid.nc'
# fh['out'] = dirs['traj'] + 'data_m' + str(mortality_ts) + '_s' + str(settling_ts) + '.pkl'
# fh['fig'] = dirs['fig'] + 'connectivity_matrix_m' + str(mortality_ts) + '_s' + str(settling_ts) + '.png'

##############################################################################
# FORMAT OUTPUT                                                              #
##############################################################################

sey = Experiment()
sey.config(dirs, preset='CMEMS', dt=timedelta(hours=1),
           releases_per_month=1, larvae_per_cell=6400)
sey.generate_dict()
sey.to_dataframe(fh='cmems6400*', parameters=parameters)

# # Generate dictionaries
# sey.generate_dict(fh['grid'],
#                                 idx_varname='coral_idx_c',
#                                 cf_varname='coral_frac_c',
#                                 cc_varname='coral_cover_c',
#                                 grp_varname='coral_grp_c',)

# seychelles_output.to_dataframe(fh['traj'], lm=lm, ls=ls, dt=3600.,
#                                lpc=6400, rpm=3, matrix=True)

# # seychelles_output.data.to_pickle(fh['out'])

# seychelles_output.export_matrix(fh['fig'], scheme='seychelles', n_years=27)
