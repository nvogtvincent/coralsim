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
pn_per_cell = 144
t0          = datetime(year=1993, month=1, day=1, hour=0)
dt          = timedelta(hours=0.5)
run_time    = timedelta(days=120)
test        = 'kernel' # Either False (no test), traj (trajectory test), kernel (kernel accuracy test)
test_params = {# Competency parameters
               'a': 1/timedelta(days=6).total_seconds(),    # Rate of competency acquisition (1/s)
               'b': 1/timedelta(days=20).total_seconds(),   # Rate of competency loss (1/s)
               'tc': timedelta(days=3.2).total_seconds(),   # Minimum competency period (s)

               # Settling parameters
               'μs': 1/timedelta(days=1).total_seconds(),   # Settling rate (1/s)

               # Mortality parameters
               'σ': -0.09,                                  # GW shape parameter 1
               'ν': 0.58,                                   # GW shape parameter 2
               'λ': 1/timedelta(days=22).total_seconds(),}  # GW mortality rate parameter

##############################################################################
# SET UP EXPERIMENT                                                          #
##############################################################################

# Create experiment object
experiment = Experiment()

# Run experiment
experiment.config(dirs, preset='CMEMS', test_params=test_params)
experiment.generate_fieldset(interp_method='freeslip')
experiment.generate_particleset(num=pn_per_cell, t0=t0, filters={'eez': [690]},
                                min_competency=timedelta(days=2),
                                dt=dt, run_time=run_time, test=test)

experiment.run(fh='example_experiment.nc')

# Analyse output
experiment.postrun_tests()



