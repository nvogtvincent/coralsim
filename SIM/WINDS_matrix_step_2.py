#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from WINDS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import xarray as xr
from glob import glob
from sys import argv

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
bio_code = argv[1]

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['input'] = dirs['root'] + 'MATRICES/STEP_1/'
dirs['output'] = dirs['root'] + 'MATRICES/'

# FILES
fh_in = sorted(glob(dirs['input'] + 'WINDS_submatrix_' + bio_code + '_*'))

##############################################################################
# COMBINE FILES                                                              #
##############################################################################

# Firstly generate the final time axis and statistics
year_list = []
day_count = 0
for fhi, fh in enumerate(fh_in):
    with xr.open_zarr(fh) as file:
        year_i = int(fh.split('_')[-1].split('.')[0])
        year_list.append(year_i)

        day_count += len(file.coords['time'])

        if fhi == 0:
            attr_dict = file.attrs
        else:
            for attr in ['a', 'b', 'configuration', 'e_num', 'interp_method',
                         'larvae_per_cell', 'max_lifespan_seconds',
                         'min_competency_seconds', 'parcels_version', 'tc',
                         'timestep_seconds', 'λ', 'μs', 'ν', 'σ']:
                assert attr_dict[attr] == file.attrs[attr]

            attr_dict['total_larvae_released'] += file.attrs['total_larvae_released']

        try:
            assert len(file.coords['time']) == 365 if year_i%4 else 366
            expected_release = attr_dict['larvae_per_cell']*8088*len(file.coords['time'])
            assert file.attrs['total_larvae_released'] == expected_release
        except:
            print('Incorrect number of particle releases in file ' + fh)
            print('Expected number of particles: ' + str(expected_release))
            print('Actual number of particles: ' + str(file.attrs['total_larvae_released']))
            difference = expected_release - file.attrs['total_larvae_released']
            print('Difference: ' + str(difference) + ' (' + str(difference/(attr_dict['larvae_per_cell']*8088)) + ' releases)')

# Check for duplicates or missing entries
year_list = np.sort(year_list)
assert (np.unique(np.sort(year_list), return_counts=True)[1] == 1).all()
assert (np.gradient(np.sort(year_list)) == 1).all()

y_start = year_list[0]
y_end = year_list[-1]
n_year = len(year_list)

# Now concatenate files
submatrix_list = []
for fhi, fh in enumerate(fh_in):
    with xr.open_zarr(fh) as file:
        submatrix_list.append(file)

full_matrix = xr.concat(submatrix_list, dim='time')
full_matrix.attrs['total_larvae_released'] = attr_dict['total_larvae_released']

assert full_matrix.attrs == attr_dict
assert day_count == len(full_matrix.coords['time'])
assert full_matrix.attrs['total_larvae_released'] == attr_dict['larvae_per_cell']*8088*len(full_matrix.coords['time'])

full_matrix.attrs['start_year'] = year_list[0]
full_matrix.attrs['end_year'] = year_list[-1]

fh_out = 'WINDS_matrix_' + bio_code + '.nc'

full_matrix.to_netcdf(dirs['output'] + fh_out)

