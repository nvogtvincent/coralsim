#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from WINDS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
from datetime import timedelta
from coralsim import Experiment
import time as timer

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
parameters = {# Competency parameters
              'a': 1/timedelta(days=5.6).total_seconds(),    # Rate of competency acquisition (1/s)
              'b': 1/timedelta(days=20).total_seconds(),     # Rate of competency loss (1/s)
              'tc': timedelta(days=3.2).total_seconds(),     # Minimum competency period (s)

              # Settling parameters
              'μs': 1/timedelta(days=1).total_seconds(),     # Settling rate (1/s)

              # Mortality parameters
              'σ': -0.09,                                    # GW shape parameter 1
              'ν': 0.58,                                     # GW shape parameter 2
              'λ': 1/timedelta(days=22.7).total_seconds(),}  # GW mortality rate parameter

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/WINDS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/'
dirs['output'] = dirs['root'] + 'MATRICES/'

##############################################################################
# FORMAT OUTPUT                                                              #
##############################################################################

sey = Experiment()
sey.config(dirs, preset='WINDS', releases_per_month=1)
sey.generate_dict()
time0 = timer.time()
matrices = sey.generate_matrix(fh='WINDS_coralsim_*',
                               parameters=parameters,
                               source_filters={'eez': [690]})

matrix_fh = dirs['output'] + 'WINDS_connectivity_matrix.nc'

matrices.to_netcdf(matrix_fh)
