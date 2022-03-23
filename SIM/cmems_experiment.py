#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run larval connectivity experiments in CMEMS
@author: Noam Vogt-Vincent
"""

import os
from datetime import datetime, timedelta
from coralsim import Experiment

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/CMEMS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/'

# PARAMETERS
pn_per_cell = 4
t0          = datetime(year=1993, month=1, day=1, hour=0)
dt          = timedelta(hours=0.1)
run_time    = timedelta(days=60)
test        = 'kernel' # Either False (no test), traj (trajectory test), kernel (kernel accuracy test)
test_params = {'a': 0.18/86400, 'b': 0.05/86400, 'tc': 3.2, 'ms': 1e-5, 'sig': -0.09, 'lam': 0.044/86400, 'nu': 0.58 }

##############################################################################
# SET UP EXPERIMENT                                                          #
##############################################################################

# Create experiment object
experiment = Experiment()

# Run experiment
experiment.config(dirs, preset='CMEMS', test_params=test_params, scheme='mp')
experiment.generate_fieldset(interp_method='freeslip')
experiment.generate_particleset(num=pn_per_cell, t0=t0, filters={'eez': [690]},
                                min_competency=timedelta(days=2),
                                dt=dt, run_time=run_time, test=test,
                                plot='grp')

experiment.run(fh='example_experiment.nc')

# Analyse output
experiment.postrun_tests()









