#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run larval connectivity experiments in CMEMS
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

# FILE HANDLES
fh = {}
fh['currents'] = sorted(glob(dirs['model'] + 'CMEMS_SFC*.nc'))
fh['grid'] = dirs['grid'] + 'coral_grid.nc'
fh['traj'] = dirs['traj'] + 'example_trajectory.nc'

# PARAMETERS
pn_per_cell = 4
t0          = datetime(year=1993, month=1, day=1, hour=0)
dt          = timedelta(hours=1)
run_time    = timedelta(days=120)
test        = True

# GRID NAMES
# dimensions_grd = {'rho': {'lon': 'lon_rho_c', 'lat': 'lat_rho_c'},
#                   'psi': {'lon': 'lon_psi_c', 'lat': 'lat_psi_c'}}
# dimensions_vel = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
# variables_vel = {'U': 'uo', 'V': 'vo'}


##############################################################################
# SET UP EXPERIMENT                                                          #
##############################################################################

# Create experiment object
experiment = Experiment()

# Import grid parameters
experiment.config(dirs, preset='CMEMS')
experiment.generate_fieldset(interp_method='freeslip')
experiment.generate_particleset(num=pn_per_cell, t0=t0, filters={'eez': [690]},
                                competency=timedelta(days=5),
                                dt=dt, run_time=run_time, test=test,
                                plot='grp', plot_fh='pos0_sey.png',)

experiment.run()




experiment.import_grid(fh=fh['grid'], grid_type='A', dimensions=dimensions_grd)

# Create particle initial states
experiment.create_particles(num_per_cell=particles_per_cell, eez_filter=[690],
                            coral_varname='coral_cover_c', grp_varname='coral_grp_c', idx_varname='coral_idx_c', eez_varname='eez_c',
                            export_coral_cover=True, export_grp=True, export_idx=True, export_eez=True,
                            plot=False, plot_colour='grp', plot_fh=dirs['fig'] + 'initial_position_seychelles.png')

# Import ocean currents (and generate FieldSet)
experiment.import_currents(fh=fh['currents'], variables=variables_vel,
                           dimensions=dimensions_vel, interp_method='freeslip')

# Add release time
experiment.add_release_time(datetime(year=release_year,
                                     month=release_month,
                                     day=release_day,
                                     hour=0))

# Add kernel fields to FieldSet
experiment.add_fields({'idx': 'coral_idx_c', 'coral_fraction': 'coral_frac_c', 'coral_cover': 'coral_cover_c'})

# Generate ParticleSet
experiment.create_particleset(fh=fh['traj'], test=False)

# Create kernels (to do: consider variable competency period)
experiment.create_kernels(competency_period=timedelta(days=5),
                          diffusion=False, dt=timedelta(minutes=60),
                          run_time=timedelta(days=120), test=False)

# Run the experiment
experiment.run()

# Carry out tests
# experiment.postrun_tests()
