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
from glob import glob
from sys import argv
from coralsim import Output

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
mortality_ts = 14
settling_ts = 1

lm = 1/(mortality_ts*24*3600) # Mortality rate (1/s)
ls = 1/(settling_ts*24*3600)  # Settling rate (1/s)

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['traj'] = dirs['root'] + 'TRAJ/'
dirs['fig'] = dirs['root'] + 'FIGURES/'

# FILE HANDLES
fh = {}
fh['traj'] = sorted(glob(dirs['traj'] + 'cmems6400*.nc'))
fh['grid'] = dirs['grid'] + 'coral_grid.nc'
fh['out'] = dirs['traj'] + 'data_m' + str(mortality_ts) + '_s' + str(settling_ts) + '.pkl'
fh['fig'] = dirs['fig'] + 'connectivity_matrix_m' + str(mortality_ts) + '_s' + str(settling_ts) + '.png'

##############################################################################
# FORMAT OUTPUT                                                              #
##############################################################################

seychelles_output = Output(dirs['root'])

# Generate dictionaries
seychelles_output.generate_dict(fh['grid'],
                                idx_varname='coral_idx_c',
                                cf_varname='coral_frac_c',
                                cc_varname='coral_cover_c',
                                grp_varname='coral_grp_c',)

seychelles_output.to_dataframe(fh['traj'], lm=lm, ls=ls, dt=3600.,
                               lpc=6400, rpm=3)

# seychelles_output.data.to_pickle(fh['out'])

seychelles_output.matrix(scheme='seychelles')
