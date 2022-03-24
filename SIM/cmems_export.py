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
import time as timer

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
parameters = {# Competency parameters
              'a': 1/timedelta(days=6).total_seconds(),    # Rate of competency acquisition (1/s)
              'b': 1/timedelta(days=20).total_seconds(),   # Rate of competency loss (1/s)
              'tc': timedelta(days=3.2).total_seconds(),   # Minimum competency period (s)

              # Settling parameters
              'μs': 1/timedelta(days=1).total_seconds(),   # Settling rate (1/s)

              # Mortality parameters
              'σ': -0.09,                                  # GW shape parameter 1
              'ν': 0.58,                                   # GW shape parameter 2
              'λ': 1/timedelta(days=22).total_seconds(),}  # GW mortality rate parameter

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
sey.config(dirs, preset='CMEMS', releases_per_month=1) ## Add ability to read from netcdf
sey.generate_dict()
time0 = timer.time()
sey.generate_matrix(fh='cmems6400*', parameters=parameters)
print(timer.time()-time0)


