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
dt          = timedelta(hours=1)
run_time    = timedelta(days=120)
test        = False # Either False (no test), traj (trajectory test), kernel (kernel accuracy test)

##############################################################################
# SET UP EXPERIMENT                                                          #
##############################################################################

# Create experiment object
experiment = Experiment()

# Run experiment
experiment.config(dirs, preset='CMEMS')
experiment.generate_fieldset(interp_method='freeslip')
experiment.generate_particleset(num=pn_per_cell, t0=t0, filters={'eez': [690]},
                                min_competency=timedelta(days=2),
                                dt=dt, run_time=run_time, test=test,
                                plot='grp', plot_fh='pos0_sey.png',)

experiment.run(fh='example_experiment.nc')

# Analyse output










