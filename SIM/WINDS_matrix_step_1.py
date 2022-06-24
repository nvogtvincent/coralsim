#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from WINDS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
import json
from datetime import timedelta
from coralsim import Experiment
from sys import argv

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
# bio_code = argv[1]
# year = argv[2]
bio_code = 'AM'
year = '1993'


# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/WINDS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/WINDS/'
dirs['output'] = dirs['root'] + 'MATRICES/'

##############################################################################
# FORMAT OUTPUT                                                              #
##############################################################################

# Get parameters from file
with open(dirs['grid'] + 'bio_parameters.json') as param_file:
    bio_data = json.load(param_file)[bio_code][0]

for key in ['a', 'b', 'μs', 'λ']:
    bio_data[key] = 1/timedelta(days=1/bio_data[key]).total_seconds()

bio_data['tc'] = timedelta(days=bio_data['tc']).total_seconds()

sey = Experiment('WINDS_' + bio_code)
sey.config(dirs, preset='WINDS', releases_per_month=1)
sey.generate_dict()
matrices = sey.generate_matrix(fh='WINDS_coralsim_' + year + '*.nc' ,
                               parameters=bio_data, figure=True)

matrix_fh = dirs['traj'] + 'STEP_1/WINDS_submatrix_' + bio_code + '_' + year + '.nc'
matrices.to_netcdf(matrix_fh)

sey.plot_parameters(fh='biological_parameters_' + bio_code + '.png')