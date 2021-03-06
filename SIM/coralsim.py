#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes and methods required to set up larval connectivity
experiments in OceanParcels.

@author: Noam Vogt-Vincent
@year: 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmasher as cmr
import pandas as pd
import cartopy.crs as ccrs
import xarray as xr
import numexpr as ne
from glob import glob
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Variable)
from netCDF4 import Dataset
from datetime import timedelta, datetime
from tqdm import tqdm


class Experiment():
    """
    Initialise a larval dispersal experiment.
    -----------
    Functions:
        # Preproc:
        config: register directories and load preset
        generate_fieldset: generate fieldsets for OceanParcels
        generate_particleset: generate initial conditions for particles + kernels

        # Runtime
        run: run OceanParcels using the above configuration

        # Postproc
        generate_dict:
        postrun_tests: Plot particle trajectories and compare larval settling
                       fluxes to the analytical offline result.

        # Not to be called:
        build_larva: build the larva class (used by generate_particleset)
        build_event: build the event kernel (used by generate_particleset)
    """


    def __init__(self, *args):
        # Set up a status dictionary so we know the completion status of the
        # experiment configuration

        self.status = {'config': False,
                       'fieldset': False,
                       'particleset': False,
                       'run': False,
                       'dict': False,
                       'matrix': False}

        try:
            self.name = args[0]
        except:
            self.name = 'my_experiment'


    def config(self, dir_dict, **kwargs):
        """
        Set directories for the script and import settings from preset

        Parameters
        ----------
        dir_dict : dictionary with 'root', 'grid', 'model', 'fig', and 'traj'
                   keys

        **kwargs : preset = 'CMEMS' or 'WINDS'

        """

        if not all (key in dir_dict.keys() for key in ['root', 'grid', 'model', 'fig', 'traj']):
            raise KeyError('Please make sure root, grid, model, fig, and traj directories have been specified.')

        if 'preset' not in kwargs.keys():
            raise NotImplementedError('Settings must currently be loaded from a preset.')

        # Note on grid types:
        # A-grid: RHO = U/V velocity defined here. Acts as 'edges' for cells.
        #         PSI = Must be calculated (as the midpoint between rho points).
        #               'Tracer' quantities (e.g. groups, coral cover, etc.)
        #               are defined here.
        # C-grid: RHO = 'Tracer' quantities (e.g. groups, coral cover, etc.)
        #               are defined here.
        #         U == (PSI, RHO) = U velocity defined here
        #         V == (RHO, PSI) = V velocity defined here

        # List all presets here
        CMEMS = {'preset': 'CMEMS',
                 'grid_filename': 'coral_grid.nc',
                 'model_filenames': 'CMEMS_SFC*.nc',

                 # Variable names for grid file
                 'grid_rc_varname' : 'reef_cover_c', # Reef cover
                 'grid_rf_varname' : 'reef_frac_c',  # Reef fraction
                 'grid_eez_varname': 'reef_eez_c',   # Reef EEZ
                 'grid_grp_varname': 'reef_grp_c',   # Reef group
                 'grid_idx_varname': 'reef_idx_c',   # Reef index,

                 # Variable types
                 'rc_dtype': np.int32,
                 'rf_dtype': np.float32,
                 'eez_dtype': np.int16,
                 'grp_dtype': np.uint8,
                 'idx_dtype': np.uint16,

                 # Dimension names for grid file
                 'lon_rho_dimname': 'lon_rho_c',
                 'lat_rho_dimname': 'lat_rho_c',
                 'lon_psi_dimname': 'lon_psi_c',
                 'lat_psi_dimname': 'lat_psi_c',

                 # Variable names for grid file
                 'u_varname': 'uo',
                 'v_varname': 'vo',

                 # Dimension names for model data
                 'lon_dimname': 'longitude',
                 'lat_dimname': 'latitude',
                 'time_dimname': 'time',

                 # Parameters for trajectory testing mode
                 'rel_lon0': 49.34,
                 'rel_lon1': 49.34,
                 'rel_lat0': -12.3,
                 'rel_lat1': -12.0,
                 'view_lon0': 47.5,
                 'view_lon1': 49.5,
                 'view_lat0': -13.5,
                 'view_lat1': -11.5,
                 'test_number': 100,
                 'lsm_varname': 'lsm_c',

                 # Grid type
                 'grid' : 'A',

                 # Maximum number of events
                 'e_num' : 36,

                 # Velocity interpolation method
                 'interp_method': 'freeslip',

                 # Plotting parameters
                 'plot': False,
                 'plot_type': 'grp',}

        WINDS = {'preset': 'WINDS',
                 'grid_filename': 'coral_grid.nc',
                 'model_filenames': 'WINDS_SFC*.nc',

                 # Variable names for grid file
                 'grid_rc_varname' : 'reef_cover_w', # Reef cover
                 'grid_rf_varname' : 'reef_frac_w',  # Reef fraction
                 'grid_eez_varname': 'reef_eez_w',   # Reef EEZ
                 'grid_grp_varname': 'reef_grp_w',   # Reef group
                 'grid_idx_varname': 'reef_idx_w',   # Reef index,

                 # Variable types
                 'rc_dtype': np.int32,
                 'rf_dtype': np.float32,
                 'eez_dtype': np.int16,
                 'grp_dtype': np.uint8,
                 'idx_dtype': np.uint16,

                 # Dimension names for grid file
                 'lon_rho_dimname': 'lon_rho_w',
                 'lat_rho_dimname': 'lat_rho_w',
                 'lon_psi_dimname': 'lon_psi_w',
                 'lat_psi_dimname': 'lat_psi_w',

                 # Variable names for grid file
                 'u_varname': 'u_surf',
                 'v_varname': 'v_surf',

                 # Dimension names for model data (psi-grid in this case)
                 'lon_dimname': 'nav_lon_u',
                 'lat_dimname': 'nav_lat_v',
                 'time_dimname': 'time_counter',

                 'lon_u_dimname': 'nav_lon_u',
                 'lat_u_dimname': 'nav_lat_u',
                 'lon_v_dimname': 'nav_lon_v',
                 'lat_v_dimname': 'nav_lat_v',

                 # Parameters for trajectory testing mode
                 'rel_lon0': 48.8,
                 'rel_lon1': 48.8,
                 'rel_lat0': -12.5,
                 'rel_lat1': -12.3,
                 'view_lon0': 48.6,
                 'view_lon1': 48.9,
                 'view_lat0': -12.5,
                 'view_lat1': -12.3,
                 'test_number': 5,
                 'lsm_varname': 'lsm_w',

                 # Grid type
                 'grid' : 'C',

                 # Maximum number of events
                 'e_num' : 60,

                 # Velocity interpolation method
                 'interp_method': 'cgrid_velocity',

                 # Plotting parameters
                 'plot': False,
                 'plot_type': 'grp',}

        PRESETS = {'CMEMS': CMEMS,
                   'WINDS': WINDS}

        if kwargs['preset'] not in PRESETS.keys():
            raise KeyError('Preset not recognised.')

        self.cfg = PRESETS[kwargs['preset']]
        self.dirs = dir_dict
        self.fh = {}

        # Further options
        if 'dt' in kwargs.keys():
            self.cfg['dt'] = kwargs['dt']

        if 'larvae_per_cell' in kwargs.keys():
            self.cfg['lpc'] = kwargs['larvae_per_cell']

        if 'releases_per_month' in kwargs.keys():
            self.cfg['rpm'] = kwargs['releases_per_month']

        if 'partitions' in kwargs.keys():
            self.cfg['partitions'] = kwargs['partitions']

        if 'test_params' in kwargs.keys():
            self.cfg['test_params'] = kwargs['test_params']


        def integrate_event_numexpr(psi0, int0, fr, a, b, tc, ??s, ??, ??, ??, t0, t1_prev, dt):

            sol = np.zeros_like(int0, dtype=np.float32)

            # Precompute reused terms
            gc_0 = ne.evaluate("b-a+(??s*fr)")
            gc_1 = ne.evaluate("??s*psi0")

            f2_gc0 = ne.evaluate("exp(t0*(b-a))")
            f3_gc0 = ne.evaluate("exp(t0*gc_0)")

            # Integrate
            for h, rk_coef in zip(np.array([0, 0.5, 1], dtype=np.float32),
                                  np.array([1/6, 2/3, 1/6], dtype=np.float32)):

                if h == 0:
                    t = t0
                else:
                    t = ne.evaluate("t0 + h*dt")

                if ?? != 0:
                    surv_t = ne.evaluate("((1. - ??*(??*(t + tc))**??)**(1/??))")
                else:
                    surv_t = ne.evaluate("exp(-(??*(t + tc))**??)")

                if h == 0:
                    f_1 = ne.evaluate("surv_t*exp(-b*t)*exp(-??s*psi0)")
                    f_3 = np.float32([0])
                else:
                    f_1 = ne.evaluate("surv_t*exp(-b*t)*exp(-??s*(psi0+fr*(t-t0)))")
                    f_3 = ne.evaluate("exp(t*gc_0) - f3_gc0")

                f_2 = ne.evaluate("f2_gc0 - exp(t1_prev*(b-a))")
                c_2 = ne.evaluate("exp(gc_1)/(b-a)")
                c_3 = ne.evaluate("exp(gc_1-(??s*fr*t0))/gc_0")

                int1 = ne.evaluate("int0 + c_2*f_2 + c_3*f_3")

                sol += ne.evaluate("rk_coef*f_1*int1")

            return ne.evaluate("a*??s*fr*dt*sol"), int1

        def integrate_event(psi0, int0, fr, a, b, tc, ??s, ??, ??, ??, t0, t1_prev, dt):

            sol = np.zeros_like(int0, dtype=np.float32)

            # Precompute reused terms
            gc_0 = b-a+(??s*fr)
            gc_1 = ??s*psi0

            f2_gc0 = np.exp(t0*(b-a))
            f3_gc0 = np.exp(t0*gc_0)

            # Integrate
            for h, rk_coef in zip(np.array([0, 0.5, 1], dtype=np.float32),
                                  np.array([1/6, 2/3, 1/6], dtype=np.float32)):

                if h == 0:
                    t = t0
                else:
                    t = t0 + h*dt

                if ?? != 0:
                    surv_t = ((1. - ??*(??*(t + tc))**??)**(1/??)).astype(np.float32)
                else:
                    surv_t = np.exp(-(??*(t + tc))**??).astype(np.float32)

                if h == 0:
                    f_1 = surv_t*np.exp(-b*t)*np.exp(-??s*psi0)
                    f_3 = np.float32([0])
                else:
                    f_1 = surv_t*np.exp(-b*t)*np.exp(-??s*(psi0+fr*(t-t0)))
                    f_3 = np.exp(t*gc_0) - f3_gc0

                f_2 = f2_gc0 - np.exp(t1_prev*(b-a))
                c_2 = np.exp(gc_1)/(b-a)
                c_3 = np.exp(gc_1-(??s*fr*t0))/gc_0

                int1 = int0 + c_2*f_2 + c_3*f_3

                sol += rk_coef*f_1*int1

            return a*??s*fr*dt*sol, int1

        self.integrate_event = integrate_event
        self.integrate_event_numexpr = integrate_event_numexpr

        self.status['config'] = True




    def generate_fieldset(self, **kwargs):
        """
        Generate the FieldSet object for OceanParcels

        """

        if not self.status['config']:
            raise Exception('Please run config first.')

        # Generate file names
        self.fh['grid'] = self.dirs['grid'] + self.cfg['grid_filename']
        self.fh['model'] = sorted(glob(self.dirs['model'] + self.cfg['model_filenames']))

        # Import grid axes
        self.axes = {}

        with Dataset(self.fh['grid'], mode='r') as nc:
            self.axes['lon_rho'] = np.array(nc.variables[self.cfg['lon_rho_dimname']][:])
            self.axes['lat_rho'] = np.array(nc.variables[self.cfg['lat_rho_dimname']][:])
            self.axes['nx_rho'] = len(self.axes['lon_rho'])
            self.axes['ny_rho'] = len(self.axes['lat_rho'])

            self.axes['lon_psi'] = np.array(nc.variables[self.cfg['lon_psi_dimname']][:])
            self.axes['lat_psi'] = np.array(nc.variables[self.cfg['lat_psi_dimname']][:])
            self.axes['nx_psi'] = len(self.axes['lon_psi'])
            self.axes['ny_psi'] = len(self.axes['lat_psi'])

        # Import currents

        if self.cfg['grid'] == 'A':
            self.fieldset = FieldSet.from_netcdf(filenames=self.fh['model'],
                                                 variables={'U': self.cfg['u_varname'],
                                                            'V': self.cfg['v_varname']},
                                                 dimensions={'U': {'lon': self.cfg['lon_dimname'],
                                                                   'lat': self.cfg['lat_dimname'],
                                                                   'time': self.cfg['time_dimname']},
                                                             'V': {'lon': self.cfg['lon_dimname'],
                                                                   'lat': self.cfg['lat_dimname'],
                                                                   'time': self.cfg['time_dimname']}},
                                                 interp_method={'U': self.cfg['interp_method'],
                                                                'V': self.cfg['interp_method']},
                                                 mesh='spherical', allow_time_extrapolation=False)
        elif self.cfg['grid'] == 'C':
            self.fieldset = FieldSet.from_nemo(filenames=self.fh['model'],
                                               variables={'U': self.cfg['u_varname'],
                                                          'V': self.cfg['v_varname']},
                                               dimensions={'U': {'lon': self.cfg['lon_dimname'],
                                                                 'lat': self.cfg['lat_dimname'],
                                                                 'time': self.cfg['time_dimname']},
                                                           'V': {'lon': self.cfg['lon_dimname'],
                                                                 'lat': self.cfg['lat_dimname'],
                                                                 'time': self.cfg['time_dimname']}},
                                               mesh='spherical', allow_time_extrapolation=False)
        else:
            raise KeyError('Grid type not understood.')

        self.fields = {}

        # Import additional fields
        if self.cfg['grid'] == 'A':
            self.field_list = ['rc', 'rf', 'eez', 'grp', 'idx']

            for field in self.field_list:
                field_varname = self.cfg['grid_' + field + '_varname']

                # Firstly verify that dimensions are correct
                with Dataset(self.fh['grid'], mode='r') as nc:
                    self.fields[field] = nc.variables[field_varname][:]

                if not np.array_equiv(np.shape(self.fields[field]),
                                      (self.axes['ny_psi'], self.axes['nx_psi'])):
                    raise Exception('Field ' + field_varname + ' has incorrect dimensions')

                if field in ['rc', 'eez', 'grp', 'idx']:
                    if np.max(self.fields[field]) > np.iinfo(self.cfg[field + '_dtype']).max:
                        raise Exception('Maximum value exceeded in ' + field_varname + '.')

                # Use OceanParcels routine to import field
                scratch_field = Field.from_netcdf(self.fh['grid'],
                                                  variable=self.cfg['grid_' + str(field) + '_varname'],
                                                  dimensions={'lon': self.cfg['lon_psi_dimname'],
                                                              'lat': self.cfg['lat_psi_dimname']},
                                                  interp_method='nearest', mesh='spherical',
                                                  allow_time_extrapolation=True)

                scratch_field.name = field
                self.fieldset.add_field(scratch_field)

        elif self.cfg['grid'] == 'C':
            self.field_list = ['rc', 'rf', 'eez', 'grp', 'idx']

            for field in self.field_list:
                field_varname = self.cfg['grid_' + field + '_varname']

                # Firstly verify that dimensions are correct
                with Dataset(self.fh['grid'], mode='r') as nc:
                    self.fields[field] = nc.variables[field_varname][:]

                if not np.array_equiv(np.shape(self.fields[field]),
                                      (self.axes['ny_rho'], self.axes['nx_rho'])):
                    raise Exception('Field ' + field_varname + ' has incorrect dimensions')

                if field in ['rc', 'eez', 'grp', 'idx']:
                    if np.max(self.fields[field]) > np.iinfo(self.cfg[field + '_dtype']).max:
                        raise Exception('Maximum value exceeded in ' + field_varname + '.')

                # Use OceanParcels routine to import field
                scratch_field = Field.from_netcdf(self.fh['grid'],
                                                  variable=self.cfg['grid_' + str(field) + '_varname'],
                                                  dimensions={'lon': self.cfg['lon_rho_dimname'],
                                                              'lat': self.cfg['lat_rho_dimname']},
                                                  interp_method='nearest', mesh='spherical',
                                                  allow_time_extrapolation=True)
                scratch_field.name = field
                self.fieldset.add_field(scratch_field)

        self.status['fieldset'] = True


    def generate_particleset(self, **kwargs):

        """
        Generate the ParticleSet object for OceanParcels

        Parameters
        ----------
        **kwargs : num = Number of particles to (aim to) release per cell
                   filters = Dict with 'eez' and/or 'grp' keys to enable filter
                             for release sites

                   t0 = Release time for particles (datetime)
                   min_competency = Minimum competency period (timedelta)
                   dt = Model time-step (timedelta)
                   run_time = Model run-time (timedelta)
                   partitions = number of partitions to split pset into
                   part = which partition to choose (1, 2...)

                   test = Whether to activate testing kernels (bool)
        """

        if not self.status['fieldset']:
            raise Exception('Please run fieldset first.')

        # Generate required default values if necessary
        if 'num' not in kwargs.keys():
            print('Particle release number not supplied.')
            print('Setting to default of 100 per cell.')
            print('')
            self.cfg['pn'] = 10
        else:
            self.cfg['pn'] = int(np.ceil(kwargs['num']**0.5))
            self.cfg['pn2'] = int(self.cfg['pn']**2)

            if np.ceil(kwargs['num']**0.5) != kwargs['num']**0.5:
                print('Particle number per cell is not square.')
                print('Old particle number: ' + str(kwargs['num']))
                print('New particle number: ' + str(self.cfg['pn2']))
                print()

            if 'lpm' in self.cfg.keys():
                if self.cfg['pn2'] != self.cfg['lpm']:
                    raise Exception('Warning: lpm is inconsistent with existing particle number setting.')

        if 't0' not in kwargs.keys():
            print('Particle release time not supplied.')
            print('Setting to default of first time in file.')
            print('')
            self.cfg['t0'] = pd.Timestamp(self.fieldset.time_origin.time_origin)
        else:
            # Check that particle release time provided is not before first
            # available time in model data
            model_start = pd.Timestamp(self.fieldset.time_origin.time_origin)
            if pd.Timestamp(kwargs['t0']) < model_start:
                print(('Particle release time has been set to ' +
                       str(pd.Timestamp(kwargs['t0'])) +
                       ' but model data starts at ' +
                       str(model_start) + '. Shifting particle start to ' +
                       str(model_start) + '.'))

                self.cfg['t0'] = model_start

            else:
                self.cfg['t0'] = kwargs['t0']

        if 'filters'  in kwargs.keys():
            for filter_name in kwargs['filters'].keys():
                if filter_name not in ['eez', 'grp']:
                    raise KeyError('Filter name ' + filter_name + ' not understood.')
            filtering = True
        else:
            filtering = False

        if 'min_competency' in kwargs.keys():
            self.cfg['min_competency'] = kwargs['min_competency']
        else:
            print('Minimum competency period not supplied.')
            print('Setting to default of 2 days.')
            print('')
            self.cfg['min_competency'] = timedelta(days=2)

        if 'dt' in kwargs.keys():
            self.cfg['dt'] = kwargs['dt']
        else:
            print('RK4 timestep not supplied.')
            print('Setting to default of 1 hour.')
            print('')
            self.cfg['dt'] = timedelta(hours=1)

        if 'run_time' in kwargs.keys():
            self.cfg['run_time'] = kwargs['run_time']
        else:
            print('Run-time not supplied.')
            print('Setting to default of 100 days.')
            print('')
            self.cfg['run_time'] = timedelta(days=100)

        if 'test' in kwargs.keys():
            if kwargs['test'] in ['kernel', 'traj', 'vis', False]:
                if kwargs['test'] in ['kernel', 'traj', 'vis']:
                    self.cfg['test'] = True
                    self.cfg['test_type'] = kwargs['test']
                else:
                    self.cfg['test'] = False
                    self.cfg['test_type'] = False
            else:
                print('Test type not understood. Ignoring test.')
                self.cfg['test'] = False
                self.cfg['test_type'] = False
        else:
            self.cfg['test'] = False
            self.cfg['test_type'] = False

        if 'partitions' in kwargs.keys():
            if 'part' in kwargs.keys():
                self.cfg['partitions'] = kwargs['partitions']
                self.cfg['part'] = kwargs['part']
            else:
                raise Exception('Please specify which part of the partitionset to release.')
        else:
            self.cfg['partitions'] = False

        # Build a mask of valid initial position cells
        reef_mask = (self.fields['rc'] > 0)
        self.cfg['nsite_nofilter'] = int(np.sum(reef_mask))

        # Filter if applicable
        if filtering:
            for filter_name in kwargs['filters'].keys():
                reef_mask *= np.isin(self.fields[filter_name], kwargs['filters'][filter_name])

        # Count number of sites identified
        self.cfg['nsite'] = int(np.sum(reef_mask))

        if self.cfg['nsite'] == 0:
            raise Exception('No valid reef sites found')
        else:
            print(str(self.cfg['nsite']) + '/' + str(self.cfg['nsite_nofilter'])  + ' reef sites identified.')
            print()

        # Find locations of sites
        reef_yidx, reef_xidx = np.where(reef_mask)

        # Generate meshgrids
        lon_rho_grid, lat_rho_grid = np.meshgrid(self.axes['lon_rho'],
                                                 self.axes['lat_rho'])
        lon_psi_grid, lat_psi_grid = np.meshgrid(self.axes['lon_psi'],
                                                 self.axes['lat_psi'])

        # Generate dictionary to hold initial particle properties
        particles = {}
        particles['lon'] = np.zeros((self.cfg['nsite']*self.cfg['pn2'],), dtype=np.float64)
        particles['lat'] = np.zeros((self.cfg['nsite']*self.cfg['pn2'],), dtype=np.float64)

        print(str(len(particles['lon'])) + ' particles generated.')
        print()

        for field in self.field_list:
            particles[field] = np.zeros((self.cfg['nsite']*self.cfg['pn2'],),
                                        dtype=self.cfg[field + '_dtype'])

        # Now evaluate each particle initial condition
        if self.cfg['grid'] == 'A':
            # For cell psi[i, j], the surrounding rho cells are:
            # rho[i, j]     (SW)
            # rho[i, j+1]   (SE)
            # rho[i+1, j]   (NW)
            # rho[i+1, j+1] (NE)

            for k, (i, j) in enumerate(zip(reef_yidx, reef_xidx)):
                # Firstly calculate the basic particle grid (may be variable for
                # curvilinear grids)

                dX = lon_rho_grid[i, j+1] - lon_rho_grid[i, j] # Grid spacing
                dY = lat_rho_grid[i+1, j] - lat_rho_grid[i, j] # Grid spacing
                dx = dX/self.cfg['pn']                         # Particle spacing
                dy = dY/self.cfg['pn']                         # Particle spacing

                gx = np.linspace(lon_rho_grid[i, j]+(dx/2),    # Particle x locations
                                 lon_rho_grid[i, j+1]-(dx/2), num=self.cfg['pn'])

                gy = np.linspace(lat_rho_grid[i, j]+(dy/2),    # Particle y locations
                                 lat_rho_grid[i+1, j]-(dy/2), num=self.cfg['pn'])

                gx, gy = [grid.flatten() for grid in np.meshgrid(gx, gy)] # Flattened arrays

                particles['lon'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gx
                particles['lat'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gy

                for field in self.field_list:
                    value_k = self.fields[field][i, j]
                    particles[field][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = value_k
        else:
            # For cell rho[i, j], the surrounding psi cells are:
            # psi[i-1, j-1] (SW)
            # psi[i-1, j]   (SE)
            # psi[i, j-1]   (NW)
            # psi[i, j]     (NE)

            for k, (i, j) in enumerate(zip(reef_yidx, reef_xidx)):
                # Firstly calculate the basic particle grid (may be variable for
                # curvilinear grids)

                dX = lon_psi_grid[i, j] - lon_psi_grid[i, j-1] # Grid spacing
                dY = lat_rho_grid[i, j] - lat_rho_grid[i-1, j] # Grid spacing
                dx = dX/self.cfg['pn']                         # Particle spacing
                dy = dY/self.cfg['pn']                         # Particle spacing

                gx = np.linspace(lon_rho_grid[i, j-1]+(dx/2),  # Particle x locations
                                 lon_rho_grid[i, j]-(dx/2), num=self.cfg['pn'])

                gy = np.linspace(lat_rho_grid[i-1, j]+(dy/2),  # Particle y locations
                                 lat_rho_grid[i, j]-(dy/2), num=self.cfg['pn'])

                gx, gy = [grid.flatten() for grid in np.meshgrid(gx, gy)] # Flattened arrays

                particles['lon'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gx
                particles['lat'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gy

                for field in self.field_list:
                    value_k = self.fields[field][i, j]
                    particles[field][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = value_k

        # Now export to DataFrame
        particles_df = pd.DataFrame({'lon': particles['lon'],
                                     'lat': particles['lat']})

        for field in self.field_list:
            particles_df[field] = particles[field]

        # Now add release times
        particles_df['t0'] = self.cfg['t0']

        # Save particles_df to class
        self.particles = particles_df

        # Set up the particle class
        if self.cfg['test_type'] != 'vis':
            self.larva = self.build_larva(self.cfg['e_num'], self.cfg['test'])
        else:
            self.larva = self.build_larva(self.cfg['e_num'], self.cfg['test'], vis=True)

        # Override for the trajectory testing mode
        if self.cfg['test_type'] == 'traj':
            # Set t0 to first time frame
            self.cfg['t0'] = model_start

            # Override all properties with a smaller testing region
            particles['lon'] = np.linspace(self.cfg['rel_lon0'],
                                           self.cfg['rel_lon1'],
                                           num=self.cfg['test_number'])
            particles['lat'] = np.linspace(self.cfg['rel_lat0'],
                                           self.cfg['rel_lat1'],
                                           num=self.cfg['test_number'])

            self.particles = pd.DataFrame({'lon': particles['lon'],
                                           'lat': particles['lat'],
                                           't0': self.cfg['t0'],
                                           'idx': 1})

        if self.cfg['partitions']:
            self.particles = np.array_split(self.particles, self.cfg['partitions'])[self.cfg['part']-1]

        # Generate the ParticleSet
        self.pset = ParticleSet.from_list(fieldset=self.fieldset,
                                          pclass=self.larva,
                                          lonlatdepth_dtype=np.float64,
                                          lon=self.particles['lon'],
                                          lat=self.particles['lat'],
                                          time=self.particles['t0'],
                                          lon0=self.particles['lon'],
                                          lat0=self.particles['lat'],
                                          idx0=self.particles['idx'])

        # Stop writing unnecessary variables
        self.pset.set_variable_write_status('depth', 'False')
        self.pset.set_variable_write_status('time', 'False')

        if not self.cfg['test']:
            self.pset.set_variable_write_status('lon', 'False')
            self.pset.set_variable_write_status('lat', 'False')

        # Add maximum age to fieldset
        self.fieldset.add_constant('max_age', int(self.cfg['run_time']/self.cfg['dt']))
        assert self.fieldset.max_age < np.iinfo(np.uint16).max

        # Add e_num to fieldset
        self.fieldset.add_constant('e_num', int(self.cfg['e_num']))

        # Add test parameters to fieldset
        if self.cfg['test']:

            param_dict = {'a': 'a', 'b': 'b', 'tc': 'tc', '??s': 'ms', '??': 'sig', '??': 'nu', '??': 'lam'}

            if 'test_params' not in self.cfg.keys():
                raise Exception('Test parameters not supplied.')

            for key in self.cfg['test_params'].keys():
                self.fieldset.add_constant(param_dict[key], self.cfg['test_params'][key])

            # In testing mode, we override the minimum competency to use tc
            self.fieldset.add_constant('min_competency', int(self.cfg['test_params']['tc']/self.cfg['dt'].total_seconds()))
        else:
            self.fieldset.add_constant('min_competency', int(self.cfg['min_competency']/self.cfg['dt']))

        # Generate kernels
        if self.cfg['test_type'] != 'vis':
            self.kernel = (self.pset.Kernel(AdvectionRK4) + self.pset.Kernel(self.build_event_kernel(self.cfg['test'])))
        else:
            self.kernel = (self.pset.Kernel(AdvectionRK4) + self.pset.Kernel(self.build_event_kernel(self.cfg['test'], vis=True)))


        # Now plot initial conditions (if wished)
        if self.cfg['plot'] and not self.cfg['test']:
            colour_series = particles[self.cfg['plot_type']]

            plot_x_range = np.max(particles['lon']) - np.min(particles['lon'])
            plot_y_range = np.max(particles['lat']) - np.min(particles['lat'])
            plot_x_range = [np.min(particles['lon']) - 0.1*plot_x_range,
                            np.max(particles['lon']) + 0.1*plot_x_range]
            plot_y_range = [np.min(particles['lat']) - 0.1*plot_y_range,
                            np.max(particles['lat']) + 0.1*plot_y_range]
            aspect = (plot_y_range[1] - plot_y_range[0])/(plot_x_range[1] - plot_x_range[0])

            f, ax = plt.subplots(1, 1, figsize=(20, 20*aspect), subplot_kw={'projection': ccrs.PlateCarree()})
            cmap = 'prism'

            ax.set_xlim(plot_x_range)
            ax.set_ylim(plot_y_range)
            ax.set_title('Initial positions for particles')

            ax.scatter(particles['lon'], particles['lat'], c=colour_series,
                       cmap=cmap, s=1, transform=ccrs.PlateCarree())

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.8, color='black', linestyle='-', zorder=11)
            gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 5))
            gl.xlocator = mticker.FixedLocator(np.arange(0, 90, 5))
            gl.xlabels_top = False
            gl.ylabels_right = False

            plt.savefig(self.dirs['fig'] + 'initial_particle_position.png', dpi=300)
            plt.close()

        self.status['particleset'] = True


    def build_larva(self, e_num, test, **kwargs):
        """
        This script builds the larva class as a test or operational class based
        on whether test is True or False

        """

        if type(test) != bool:
            raise Exception('Input must be a boolean.')

        if 'vis' in kwargs:
            vis_mode = kwargs['vis']
        else:
            vis_mode = False

        if test and not vis_mode:
            class larva(JITParticle):

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                ##################################################################

                # idx of current cell (>0 if in any reef cell)
                idx = Variable('idx',
                               dtype=np.int32,
                               initial=0,
                               to_write=True)

                # Time at sea (Total time steps since spawning)
                ot  = Variable('ot',
                               dtype=np.int32,
                               initial=0,
                               to_write=True)

                # Active status
                active = Variable('active',
                                  dtype=np.uint8,
                                  initial=1,
                                  to_write=False)

                ##################################################################
                # PROVENANCE IDENTIFIERS #########################################
                ##################################################################

                # Group of parent reef
                idx0 = Variable('idx0',
                                dtype=np.int32,
                                to_write=True)

                # Original longitude
                lon0 = Variable('lon0',
                                dtype=np.float32,
                                to_write=True)

                # Original latitude
                lat0 = Variable('lat0',
                                dtype=np.float32,
                                to_write=True)

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING SETTLING AT REEF SITES ########
                ##################################################################

                # Current reef time (record of timesteps spent at current reef cell)
                # Switch to uint16 if possible!
                current_reef_ts = Variable('current_reef_ts',
                                           dtype=np.int16,
                                           initial=0,
                                           to_write=False)

                # Current reef t0 (record of arrival time (in timesteps) at current reef)
                # Switch to uint16 if possible!
                current_reef_ts0 = Variable('current_reef_ts0',
                                            dtype=np.int16,
                                            initial=0,
                                            to_write=False)

                # Current reef idx (record of the index of the current reef
                # Switch to uint16 if possible!
                current_reef_idx = Variable('current_reef_idx',
                                            dtype=np.int32,
                                            initial=0.,
                                            to_write=False)

                ##################################################################
                # RECORD OF ALL EVENTS ###########################################
                ##################################################################

                # Number of events
                e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=True)

                # Event variables (i = idx, t = arrival time(step), dt = time(steps) at reef)
                # are added dynamically

                ##################################################################
                # TEMPORARY TESTING VARIABLES ####################################
                ##################################################################

                # Number of larvae accumulated in the current reef
                L1 = Variable('L1', dtype=np.float64, initial=1., to_write=True) # Pre-competent larvae
                L2 = Variable('L2', dtype=np.float64, initial=0., to_write=True) # Competent larvae
                L10 = Variable('L10', dtype=np.float64, initial=0., to_write=True) # Pre-competent larvae, frozen at start
                L20 = Variable('L20', dtype=np.float64, initial=0., to_write=True) # Competent larvae, frozen at start
                Ns = Variable('Ns', dtype=np.float64, initial=0., to_write=True) # Larvae settling in current/just-passed event
                Ns_next = Variable('Ns_next', dtype=np.float64, initial=0., to_write=True) # Larvae settling in current event (when event has just ended)

                # Reef fraction
                rf = Variable('rf', dtype=np.float32, initial=0., to_write=True)

                # Mortality coefficient mu_m
                mm = Variable('mm', dtype=np.float64, initial=0., to_write=True)

        elif test:
            class larva(JITParticle):

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                ##################################################################

                # idx of current cell (>0 if in any reef cell)
                idx = Variable('idx',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Time at sea (Total time steps since spawning)
                ot  = Variable('ot',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Active status
                active = Variable('active',
                                  dtype=np.uint8,
                                  initial=1,
                                  to_write=False)

                ##################################################################
                # PROVENANCE IDENTIFIERS #########################################
                ##################################################################

                # Group of parent reef
                idx0 = Variable('idx0',
                                dtype=np.int32,
                                to_write=False)

                # Original longitude
                lon0 = Variable('lon0',
                                dtype=np.float32,
                                to_write=False)

                # Original latitude
                lat0 = Variable('lat0',
                                dtype=np.float32,
                                to_write=False)

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING SETTLING AT REEF SITES ########
                ##################################################################

                # Current reef time (record of timesteps spent at current reef cell)
                # Switch to uint16 if possible!
                current_reef_ts = Variable('current_reef_ts',
                                           dtype=np.int16,
                                           initial=0,
                                           to_write=False)

                # Current reef t0 (record of arrival time (in timesteps) at current reef)
                # Switch to uint16 if possible!
                current_reef_ts0 = Variable('current_reef_ts0',
                                            dtype=np.int16,
                                            initial=0,
                                            to_write=False)

                # Current reef idx (record of the index of the current reef
                # Switch to uint16 if possible!
                current_reef_idx = Variable('current_reef_idx',
                                            dtype=np.int32,
                                            initial=0.,
                                            to_write=False)

                ##################################################################
                # RECORD OF ALL EVENTS ###########################################
                ##################################################################

                # Number of events
                e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=False)

                # Event variables (i = idx, t = arrival time(step), dt = time(steps) at reef)
                # are added dynamically

                ##################################################################
                # TEMPORARY TESTING VARIABLES ####################################
                ##################################################################

                # Number of larvae accumulated in the current reef
                L1 = Variable('L1', dtype=np.float64, initial=1., to_write=False) # Pre-competent larvae
                L2 = Variable('L2', dtype=np.float64, initial=0., to_write=True) # Competent larvae
                L10 = Variable('L10', dtype=np.float64, initial=0., to_write=False) # Pre-competent larvae, frozen at start
                L20 = Variable('L20', dtype=np.float64, initial=0., to_write=False) # Competent larvae, frozen at start
                Ns = Variable('Ns', dtype=np.float64, initial=0., to_write=False) # Larvae settling in current/just-passed event
                Ns_next = Variable('Ns_next', dtype=np.float64, initial=0., to_write=False) # Larvae settling in current event (when event has just ended)

                # Reef fraction
                rf = Variable('rf', dtype=np.float32, initial=0., to_write=False)

                # Mortality coefficient mu_m
                mm = Variable('mm', dtype=np.float64, initial=0., to_write=False)

        else:
            class larva(JITParticle):

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                ##################################################################

                # idx of current cell (>0 if in any reef cell)
                idx = Variable('idx',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Time at sea (Total time steps since spawning)
                ot  = Variable('ot',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Active status
                active = Variable('active',
                                  dtype=np.uint8,
                                  initial=1,
                                  to_write=False)

                ##################################################################
                # PROVENANCE IDENTIFIERS #########################################
                ##################################################################

                # Group of parent reef
                idx0 = Variable('idx0',
                                dtype=np.uint16,
                                to_write=True)

                # Original longitude
                lon0 = Variable('lon0',
                                dtype=np.float32,
                                to_write=True)

                # Original latitude
                lat0 = Variable('lat0',
                                dtype=np.float32,
                                to_write=True)

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING SETTLING AT REEF SITES ########
                ##################################################################

                # Current reef time (record of timesteps spent at current reef cell)
                # Switch to uint16 if possible!
                current_reef_ts = Variable('current_reef_ts',
                                           dtype=np.uint16,
                                           initial=0,
                                           to_write=False)

                # Current reef t0 (record of arrival time (in timesteps) at current reef)
                # Switch to uint16 if possible!
                current_reef_ts0 = Variable('current_reef_ts0',
                                            dtype=np.uint16,
                                            initial=0,
                                            to_write=False)

                # Current reef idx (record of the index of the current reef
                # Switch to uint16 if possible!
                current_reef_idx = Variable('current_reef_idx',
                                            dtype=np.uint16,
                                            initial=0.,
                                            to_write=False)

                ##################################################################
                # RECORD OF ALL EVENTS ###########################################
                ##################################################################

                # Number of events
                e_num = Variable('e_num', dtype=np.uint8, initial=0, to_write=True)

                # Event variables (i = idx, t = arrival time(step), dt = time(steps) at reef)
                # are added dynamically

        if not test or not vis_mode:
            for e_val in range(e_num):
                setattr(larva, 'i' + str(e_val), Variable('i' + str(e_val), dtype=np.uint16, initial=0, to_write=True))
                setattr(larva, 'ts' + str(e_val), Variable('ts' + str(e_val), dtype=np.uint16, initial=0, to_write=True))
                setattr(larva, 'dt' + str(e_val), Variable('dt' + str(e_val), dtype=np.uint16, initial=0, to_write=True))

                if test:
                    setattr(larva, 'Ns' + str(e_val), Variable('Ns' + str(e_val), dtype=np.float32, initial=0., to_write=True))

        return larva

    def build_event_kernel(self, test, **kwargs):
        """
        This script builds the event kernel as a test or operational kernel based
        on whether test is True or False

        """

        if type(test) != bool:
            raise Exception('Input must be a boolean.')

        if 'vis' in kwargs:
            vis_mode = kwargs['vis']
        else:
            vis_mode = False

        if test and not vis_mode:
            # STANDARD TEST KERNEL
            def event(particle, fieldset, time):

                # 1 Keep track of the amount of time spent at sea
                particle.ot += 1

                ###############################################################
                # ACTIVE PARTICLES ONLY                                       #
                ###############################################################

                if particle.active:

                    # 2 Assess reef status
                    particle.idx = fieldset.idx[particle]

                    # TESTING ONLY ############################################
                    # Calculate current mortality rate
                    particle.mm = (fieldset.lam*fieldset.nu)*((fieldset.lam*particle.ot*particle.dt)**(fieldset.nu-1))/(1-fieldset.sig*((fieldset.lam*particle.ot*particle.dt)**fieldset.nu))
                    particle.L10 = particle.L1
                    particle.L20 = particle.L2

                    particle.rf = fieldset.rf[particle]
                    ###########################################################

                    save_event = False
                    new_event = False

                    # 3 Trigger event cascade if larva is in a reef site and minimum competency has been reached
                    if particle.idx > 0 and particle.ot > fieldset.min_competency:

                        # Check if an event has already been triggered
                        if particle.current_reef_ts > 0:

                            # Check if we are in the same reef idx as the current event
                            if particle.idx == particle.current_reef_idx:

                                # If contiguous event, just add time and phi
                                particle.current_reef_ts += 1

                                # TESTING ONLY ############################################
                                particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                                particle.Ns = particle.Ns + fieldset.ms*particle.rf*particle.L20*particle.dt
                                ###########################################################

                                # But also check that the particle isn't about to expire (save if so)
                                # Otherwise particles hanging around reefs at the end of the simulation
                                # won't get saved.

                                if particle.ot > fieldset.max_age:
                                    save_event = True

                            else:

                                # TESTING ONLY ############################################
                                particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                                particle.Ns = particle.Ns
                                particle.Ns_next = fieldset.ms*particle.rf*particle.L20*particle.dt
                                ###########################################################

                                # Otherwise, we need to save the old event and create a new event
                                save_event = True
                                new_event = True

                        else:

                            # TESTING ONLY ############################################
                            particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                            particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                            particle.Ns_next = fieldset.ms*particle.rf*particle.L20*particle.dt
                            ###########################################################

                            # If event has not been triggered, create a new event
                            new_event = True

                    else:

                        # Otherwise, check if ongoing event has just ended
                        if particle.current_reef_ts > 0 and particle.ot > fieldset.min_competency:

                            # TESTING ONLY ############################################
                            particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                            particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                            particle.Ns = particle.Ns + fieldset.ms*particle.rf*particle.L20*particle.dt
                            ###########################################################

                            save_event = True

                        elif particle.ot > fieldset.min_competency:
                            # TESTING ONLY ############################################
                            particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                            particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm)*particle.L20)*particle.dt
                            ###########################################################

                        else:
                            # TESTING ONLY ############################################
                            particle.L1 = particle.L10 - (particle.mm)*particle.L10*particle.dt
                            ###########################################################


                    if save_event:
                        # Save current values
                        # Unfortunately since setattr doesn't work in a kernel, this
                        # requires a horrendous elif chain.

                        if particle.e_num == 0:
                            particle.i0 = particle.current_reef_idx
                            particle.ts0 = particle.current_reef_ts0
                            particle.dt0 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns0 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 1:
                            particle.i1 = particle.current_reef_idx
                            particle.ts1 = particle.current_reef_ts0
                            particle.dt1 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns1 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 2:
                            particle.i2 = particle.current_reef_idx
                            particle.ts2 = particle.current_reef_ts0
                            particle.dt2 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns2 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 3:
                            particle.i3 = particle.current_reef_idx
                            particle.ts3 = particle.current_reef_ts0
                            particle.dt3 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns3 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 4:
                            particle.i4 = particle.current_reef_idx
                            particle.ts4 = particle.current_reef_ts0
                            particle.dt4 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns4 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 5:
                            particle.i5 = particle.current_reef_idx
                            particle.ts5 = particle.current_reef_ts0
                            particle.dt5 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns5 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 6:
                            particle.i6 = particle.current_reef_idx
                            particle.ts6 = particle.current_reef_ts0
                            particle.dt6 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns6 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 7:
                            particle.i7 = particle.current_reef_idx
                            particle.ts7 = particle.current_reef_ts0
                            particle.dt7 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns7 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 8:
                            particle.i8 = particle.current_reef_idx
                            particle.ts8 = particle.current_reef_ts0
                            particle.dt8 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns8 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 9:
                            particle.i9 = particle.current_reef_idx
                            particle.ts9 = particle.current_reef_ts0
                            particle.dt9 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns9 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 10:
                            particle.i10 = particle.current_reef_idx
                            particle.ts10 = particle.current_reef_ts0
                            particle.dt10 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns10 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 11:
                            particle.i11 = particle.current_reef_idx
                            particle.ts11 = particle.current_reef_ts0
                            particle.dt11 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns11 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 12:
                            particle.i12 = particle.current_reef_idx
                            particle.ts12 = particle.current_reef_ts0
                            particle.dt12 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns12 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 13:
                            particle.i13 = particle.current_reef_idx
                            particle.ts13 = particle.current_reef_ts0
                            particle.dt13 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns13 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 14:
                            particle.i14 = particle.current_reef_idx
                            particle.ts14 = particle.current_reef_ts0
                            particle.dt14 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns14 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 15:
                            particle.i15 = particle.current_reef_idx
                            particle.ts15 = particle.current_reef_ts0
                            particle.dt15 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns15 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 16:
                            particle.i16 = particle.current_reef_idx
                            particle.ts16 = particle.current_reef_ts0
                            particle.dt16 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns16 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 17:
                            particle.i17 = particle.current_reef_idx
                            particle.ts17 = particle.current_reef_ts0
                            particle.dt17 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns17 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 18:
                            particle.i18 = particle.current_reef_idx
                            particle.ts18 = particle.current_reef_ts0
                            particle.dt18 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns18 = particle.Ns
                            ###########################################################
                        elif particle.e_num == 19:
                            particle.i19 = particle.current_reef_idx
                            particle.ts19 = particle.current_reef_ts0
                            particle.dt19 = particle.current_reef_ts
                            # TESTING ONLY ############################################
                            particle.Ns19 = particle.Ns
                            ###########################################################

                            particle.active = 0 # Deactivate particle, since no more reefs can be saved

                        # Then reset current values to zero
                        particle.current_reef_idx = 0
                        particle.current_reef_ts0 = 0
                        particle.current_reef_ts = 0

                        # Add to event number counter
                        particle.e_num += 1

                    if new_event:
                        # Add status to current (for current event) values
                        # Timesteps at current reef
                        particle.current_reef_ts = 1

                        # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                        particle.current_reef_ts0 = particle.ot - 1

                        # Current reef group
                        particle.current_reef_idx = particle.idx

                        # TESTING ONLY ############################################
                        particle.Ns = particle.Ns_next
                        ###########################################################

                # Finally, check if particle needs to be deleted
                if particle.ot >= fieldset.max_age:

                    # Only delete particles where at least 1 event has been recorded
                    if particle.e_num > 0:
                        particle.delete()

        elif test:
            # VISUALISATION KERNEL
            def event(particle, fieldset, time):

                # 1 Keep track of the amount of time spent at sea
                particle.ot += 1

                # 2 Assess reef status
                particle.idx = fieldset.idx[particle]

                # Calculate current mortality rate
                particle.mm = (fieldset.lam*fieldset.nu)*((fieldset.lam*particle.ot*particle.dt)**(fieldset.nu-1))/(1-fieldset.sig*((fieldset.lam*particle.ot*particle.dt)**fieldset.nu))
                particle.L10 = particle.L1
                particle.L20 = particle.L2

                particle.rf = fieldset.rf[particle]

                # 3 Trigger event cascade if larva is in a reef site and minimum competency has been reached
                if particle.idx > 0 and particle.ot > fieldset.min_competency:

                    particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                    particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt

                elif particle.ot > fieldset.min_competency:

                    particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                    particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm)*particle.L20)*particle.dt

                else:

                    particle.L1 = particle.L10 - (particle.mm)*particle.L10*particle.dt

        else:
            # OPERATIONAL CMEMS KERNEL
            if self.cfg['preset'] == 'CMEMS':
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    ###############################################################
                    # ACTIVE PARTICLES ONLY                                       #
                    ###############################################################

                    if particle.active:

                        # 2 Assess reef status
                        particle.idx = fieldset.idx[particle]

                        save_event = False
                        new_event = False

                        # 3 Trigger event cascade if larva is in a reef site and competency has been reached
                        if particle.idx > 0 and particle.ot > fieldset.min_competency:

                            # Check if an event has already been triggered
                            if particle.current_reef_ts > 0:

                                # Check if we are in the same reef idx as the current event
                                if particle.idx == particle.current_reef_idx:

                                    # If contiguous event, just add time and phi
                                    particle.current_reef_ts += 1

                                    # But also check that the particle isn't about to expire (save if so)
                                    # Otherwise particles hanging around reefs at the end of the simulation
                                    # won't get saved.

                                    if particle.ot > fieldset.max_age:
                                        save_event = True

                                else:

                                    # Otherwise, we need to save the old event and create a new event
                                    save_event = True
                                    new_event = True

                            else:

                                # If event has not been triggered, create a new event
                                new_event = True

                        else:

                            # Otherwise, check if ongoing event has just ended
                            if particle.current_reef_ts > 0:

                                save_event = True

                        if save_event:
                            # Save current values
                            # Unfortunately, due to the limited functions allowed in parcels, this
                            # required an horrendous if-else chain

                            if particle.e_num == 0:
                                particle.i0 = particle.current_reef_idx
                                particle.ts0 = particle.current_reef_ts0
                                particle.dt0 = particle.current_reef_ts
                            elif particle.e_num == 1:
                                particle.i1 = particle.current_reef_idx
                                particle.ts1 = particle.current_reef_ts0
                                particle.dt1 = particle.current_reef_ts
                            elif particle.e_num == 2:
                                particle.i2 = particle.current_reef_idx
                                particle.ts2 = particle.current_reef_ts0
                                particle.dt2 = particle.current_reef_ts
                            elif particle.e_num == 3:
                                particle.i3 = particle.current_reef_idx
                                particle.ts3 = particle.current_reef_ts0
                                particle.dt3 = particle.current_reef_ts
                            elif particle.e_num == 4:
                                particle.i4 = particle.current_reef_idx
                                particle.ts4 = particle.current_reef_ts0
                                particle.dt4 = particle.current_reef_ts
                            elif particle.e_num == 5:
                                particle.i5 = particle.current_reef_idx
                                particle.ts5 = particle.current_reef_ts0
                                particle.dt5 = particle.current_reef_ts
                            elif particle.e_num == 6:
                                particle.i6 = particle.current_reef_idx
                                particle.ts6 = particle.current_reef_ts0
                                particle.dt6 = particle.current_reef_ts
                            elif particle.e_num == 7:
                                particle.i7 = particle.current_reef_idx
                                particle.ts7 = particle.current_reef_ts0
                                particle.dt7 = particle.current_reef_ts
                            elif particle.e_num == 8:
                                particle.i8 = particle.current_reef_idx
                                particle.ts8 = particle.current_reef_ts0
                                particle.dt8 = particle.current_reef_ts
                            elif particle.e_num == 9:
                                particle.i9 = particle.current_reef_idx
                                particle.ts9 = particle.current_reef_ts0
                                particle.dt9 = particle.current_reef_ts
                            elif particle.e_num == 10:
                                particle.i10 = particle.current_reef_idx
                                particle.ts10 = particle.current_reef_ts0
                                particle.dt10 = particle.current_reef_ts
                            elif particle.e_num == 11:
                                particle.i11 = particle.current_reef_idx
                                particle.ts11 = particle.current_reef_ts0
                                particle.dt11 = particle.current_reef_ts
                            elif particle.e_num == 12:
                                particle.i12 = particle.current_reef_idx
                                particle.ts12 = particle.current_reef_ts0
                                particle.dt12 = particle.current_reef_ts
                            elif particle.e_num == 13:
                                particle.i13 = particle.current_reef_idx
                                particle.ts13 = particle.current_reef_ts0
                                particle.dt13 = particle.current_reef_ts
                            elif particle.e_num == 14:
                                particle.i14 = particle.current_reef_idx
                                particle.ts14 = particle.current_reef_ts0
                                particle.dt14 = particle.current_reef_ts
                            elif particle.e_num == 15:
                                particle.i15 = particle.current_reef_idx
                                particle.ts15 = particle.current_reef_ts0
                                particle.dt15 = particle.current_reef_ts
                            elif particle.e_num == 16:
                                particle.i16 = particle.current_reef_idx
                                particle.ts16 = particle.current_reef_ts0
                                particle.dt16 = particle.current_reef_ts
                            elif particle.e_num == 17:
                                particle.i17 = particle.current_reef_idx
                                particle.ts17 = particle.current_reef_ts0
                                particle.dt17 = particle.current_reef_ts
                            elif particle.e_num == 18:
                                particle.i18 = particle.current_reef_idx
                                particle.ts18 = particle.current_reef_ts0
                                particle.dt18 = particle.current_reef_ts
                            elif particle.e_num == 19:
                                particle.i19 = particle.current_reef_idx
                                particle.ts19 = particle.current_reef_ts0
                                particle.dt19 = particle.current_reef_ts
                            elif particle.e_num == 20:
                                particle.i20 = particle.current_reef_idx
                                particle.ts20 = particle.current_reef_ts0
                                particle.dt20 = particle.current_reef_ts
                            elif particle.e_num == 21:
                                particle.i21 = particle.current_reef_idx
                                particle.ts21 = particle.current_reef_ts0
                                particle.dt21 = particle.current_reef_ts
                            elif particle.e_num == 22:
                                particle.i22 = particle.current_reef_idx
                                particle.ts22 = particle.current_reef_ts0
                                particle.dt22 = particle.current_reef_ts
                            elif particle.e_num == 23:
                                particle.i23 = particle.current_reef_idx
                                particle.ts23 = particle.current_reef_ts0
                                particle.dt23 = particle.current_reef_ts
                            elif particle.e_num == 24:
                                particle.i24 = particle.current_reef_idx
                                particle.ts24 = particle.current_reef_ts0
                                particle.dt24 = particle.current_reef_ts
                            elif particle.e_num == 25:
                                particle.i25 = particle.current_reef_idx
                                particle.ts25 = particle.current_reef_ts0
                                particle.dt25 = particle.current_reef_ts
                            elif particle.e_num == 26:
                                particle.i26 = particle.current_reef_idx
                                particle.ts26 = particle.current_reef_ts0
                                particle.dt26 = particle.current_reef_ts
                            elif particle.e_num == 27:
                                particle.i27 = particle.current_reef_idx
                                particle.ts27 = particle.current_reef_ts0
                                particle.dt27 = particle.current_reef_ts
                            elif particle.e_num == 28:
                                particle.i28 = particle.current_reef_idx
                                particle.ts28 = particle.current_reef_ts0
                                particle.dt28 = particle.current_reef_ts
                            elif particle.e_num == 29:
                                particle.i29 = particle.current_reef_idx
                                particle.ts29 = particle.current_reef_ts0
                                particle.dt29 = particle.current_reef_ts
                            elif particle.e_num == 30:
                                particle.i30 = particle.current_reef_idx
                                particle.ts30 = particle.current_reef_ts0
                                particle.dt30 = particle.current_reef_ts
                            elif particle.e_num == 31:
                                particle.i31 = particle.current_reef_idx
                                particle.ts31 = particle.current_reef_ts0
                                particle.dt31 = particle.current_reef_ts
                            elif particle.e_num == 32:
                                particle.i32 = particle.current_reef_idx
                                particle.ts32 = particle.current_reef_ts0
                                particle.dt32 = particle.current_reef_ts
                            elif particle.e_num == 33:
                                particle.i33 = particle.current_reef_idx
                                particle.ts33 = particle.current_reef_ts0
                                particle.dt33 = particle.current_reef_ts
                            elif particle.e_num == 34:
                                particle.i34 = particle.current_reef_idx
                                particle.ts34 = particle.current_reef_ts0
                                particle.dt34 = particle.current_reef_ts
                            elif particle.e_num == 35:
                                particle.i35 = particle.current_reef_idx
                                particle.ts35 = particle.current_reef_ts0
                                particle.dt35 = particle.current_reef_ts

                                particle.active = 0  # Deactivate particle, since no more reefs can be saved

                            # Then reset current values to zero
                            particle.current_reef_idx = 0
                            particle.current_reef_ts0 = 0
                            particle.current_reef_ts = 0

                            # Add to event number counter
                            particle.e_num += 1

                        if new_event:
                            # Add status to current (for current event) values
                            # Timesteps at current reef
                            particle.current_reef_ts = 1

                            # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                            particle.current_reef_ts0 = particle.ot - 1

                            # Current reef group
                            particle.current_reef_idx = particle.idx

                    # Finally, check if particle needs to be deleted
                    if particle.ot >= fieldset.max_age:

                        # Only delete particles where at least 1 event has been recorded
                        if particle.e_num > 0:
                            particle.delete()

            elif self.cfg['preset'] == 'WINDS':
                # OPERATIONAL WINDS KERNEL
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    ###############################################################
                    # ACTIVE PARTICLES ONLY                                       #
                    ###############################################################

                    if particle.active:

                        # 2 Assess reef status
                        particle.idx = fieldset.idx[particle]

                        save_event = False
                        new_event = False

                        # 3 Trigger event cascade if larva is in a reef site and competency has been reached
                        if particle.idx > 0 and particle.ot > fieldset.min_competency:

                            # Check if an event has already been triggered
                            if particle.current_reef_ts > 0:

                                # Check if we are in the same reef idx as the current event
                                if particle.idx == particle.current_reef_idx:

                                    # If contiguous event, just add time and phi
                                    particle.current_reef_ts += 1

                                    # But also check that the particle isn't about to expire (save if so)
                                    # Otherwise particles hanging around reefs at the end of the simulation
                                    # won't get saved.

                                    if particle.ot > fieldset.max_age:
                                        save_event = True

                                else:

                                    # Otherwise, we need to save the old event and create a new event
                                    save_event = True
                                    new_event = True

                            else:

                                # If event has not been triggered, create a new event
                                new_event = True

                        else:

                            # Otherwise, check if ongoing event has just ended
                            if particle.current_reef_ts > 0:

                                save_event = True

                        if save_event:
                            # Save current values
                            # Unfortunately, due to the limited functions allowed in parcels, this
                            # required an horrendous if-else chain

                            if particle.e_num == 0:
                                particle.i0 = particle.current_reef_idx
                                particle.ts0 = particle.current_reef_ts0
                                particle.dt0 = particle.current_reef_ts
                            elif particle.e_num == 1:
                                particle.i1 = particle.current_reef_idx
                                particle.ts1 = particle.current_reef_ts0
                                particle.dt1 = particle.current_reef_ts
                            elif particle.e_num == 2:
                                particle.i2 = particle.current_reef_idx
                                particle.ts2 = particle.current_reef_ts0
                                particle.dt2 = particle.current_reef_ts
                            elif particle.e_num == 3:
                                particle.i3 = particle.current_reef_idx
                                particle.ts3 = particle.current_reef_ts0
                                particle.dt3 = particle.current_reef_ts
                            elif particle.e_num == 4:
                                particle.i4 = particle.current_reef_idx
                                particle.ts4 = particle.current_reef_ts0
                                particle.dt4 = particle.current_reef_ts
                            elif particle.e_num == 5:
                                particle.i5 = particle.current_reef_idx
                                particle.ts5 = particle.current_reef_ts0
                                particle.dt5 = particle.current_reef_ts
                            elif particle.e_num == 6:
                                particle.i6 = particle.current_reef_idx
                                particle.ts6 = particle.current_reef_ts0
                                particle.dt6 = particle.current_reef_ts
                            elif particle.e_num == 7:
                                particle.i7 = particle.current_reef_idx
                                particle.ts7 = particle.current_reef_ts0
                                particle.dt7 = particle.current_reef_ts
                            elif particle.e_num == 8:
                                particle.i8 = particle.current_reef_idx
                                particle.ts8 = particle.current_reef_ts0
                                particle.dt8 = particle.current_reef_ts
                            elif particle.e_num == 9:
                                particle.i9 = particle.current_reef_idx
                                particle.ts9 = particle.current_reef_ts0
                                particle.dt9 = particle.current_reef_ts
                            elif particle.e_num == 10:
                                particle.i10 = particle.current_reef_idx
                                particle.ts10 = particle.current_reef_ts0
                                particle.dt10 = particle.current_reef_ts
                            elif particle.e_num == 11:
                                particle.i11 = particle.current_reef_idx
                                particle.ts11 = particle.current_reef_ts0
                                particle.dt11 = particle.current_reef_ts
                            elif particle.e_num == 12:
                                particle.i12 = particle.current_reef_idx
                                particle.ts12 = particle.current_reef_ts0
                                particle.dt12 = particle.current_reef_ts
                            elif particle.e_num == 13:
                                particle.i13 = particle.current_reef_idx
                                particle.ts13 = particle.current_reef_ts0
                                particle.dt13 = particle.current_reef_ts
                            elif particle.e_num == 14:
                                particle.i14 = particle.current_reef_idx
                                particle.ts14 = particle.current_reef_ts0
                                particle.dt14 = particle.current_reef_ts
                            elif particle.e_num == 15:
                                particle.i15 = particle.current_reef_idx
                                particle.ts15 = particle.current_reef_ts0
                                particle.dt15 = particle.current_reef_ts
                            elif particle.e_num == 16:
                                particle.i16 = particle.current_reef_idx
                                particle.ts16 = particle.current_reef_ts0
                                particle.dt16 = particle.current_reef_ts
                            elif particle.e_num == 17:
                                particle.i17 = particle.current_reef_idx
                                particle.ts17 = particle.current_reef_ts0
                                particle.dt17 = particle.current_reef_ts
                            elif particle.e_num == 18:
                                particle.i18 = particle.current_reef_idx
                                particle.ts18 = particle.current_reef_ts0
                                particle.dt18 = particle.current_reef_ts
                            elif particle.e_num == 19:
                                particle.i19 = particle.current_reef_idx
                                particle.ts19 = particle.current_reef_ts0
                                particle.dt19 = particle.current_reef_ts
                            elif particle.e_num == 20:
                                particle.i20 = particle.current_reef_idx
                                particle.ts20 = particle.current_reef_ts0
                                particle.dt20 = particle.current_reef_ts
                            elif particle.e_num == 21:
                                particle.i21 = particle.current_reef_idx
                                particle.ts21 = particle.current_reef_ts0
                                particle.dt21 = particle.current_reef_ts
                            elif particle.e_num == 22:
                                particle.i22 = particle.current_reef_idx
                                particle.ts22 = particle.current_reef_ts0
                                particle.dt22 = particle.current_reef_ts
                            elif particle.e_num == 23:
                                particle.i23 = particle.current_reef_idx
                                particle.ts23 = particle.current_reef_ts0
                                particle.dt23 = particle.current_reef_ts
                            elif particle.e_num == 24:
                                particle.i24 = particle.current_reef_idx
                                particle.ts24 = particle.current_reef_ts0
                                particle.dt24 = particle.current_reef_ts
                            elif particle.e_num == 25:
                                particle.i25 = particle.current_reef_idx
                                particle.ts25 = particle.current_reef_ts0
                                particle.dt25 = particle.current_reef_ts
                            elif particle.e_num == 26:
                                particle.i26 = particle.current_reef_idx
                                particle.ts26 = particle.current_reef_ts0
                                particle.dt26 = particle.current_reef_ts
                            elif particle.e_num == 27:
                                particle.i27 = particle.current_reef_idx
                                particle.ts27 = particle.current_reef_ts0
                                particle.dt27 = particle.current_reef_ts
                            elif particle.e_num == 28:
                                particle.i28 = particle.current_reef_idx
                                particle.ts28 = particle.current_reef_ts0
                                particle.dt28 = particle.current_reef_ts
                            elif particle.e_num == 29:
                                particle.i29 = particle.current_reef_idx
                                particle.ts29 = particle.current_reef_ts0
                                particle.dt29 = particle.current_reef_ts
                            elif particle.e_num == 30:
                                particle.i30 = particle.current_reef_idx
                                particle.ts30 = particle.current_reef_ts0
                                particle.dt30 = particle.current_reef_ts
                            elif particle.e_num == 31:
                                particle.i31 = particle.current_reef_idx
                                particle.ts31 = particle.current_reef_ts0
                                particle.dt31 = particle.current_reef_ts
                            elif particle.e_num == 32:
                                particle.i32 = particle.current_reef_idx
                                particle.ts32 = particle.current_reef_ts0
                                particle.dt32 = particle.current_reef_ts
                            elif particle.e_num == 33:
                                particle.i33 = particle.current_reef_idx
                                particle.ts33 = particle.current_reef_ts0
                                particle.dt33 = particle.current_reef_ts
                            elif particle.e_num == 34:
                                particle.i34 = particle.current_reef_idx
                                particle.ts34 = particle.current_reef_ts0
                                particle.dt34 = particle.current_reef_ts
                            elif particle.e_num == 35:
                                particle.i35 = particle.current_reef_idx
                                particle.ts35 = particle.current_reef_ts0
                                particle.dt35 = particle.current_reef_ts
                            elif particle.e_num == 36:
                                particle.i36 = particle.current_reef_idx
                                particle.ts36 = particle.current_reef_ts0
                                particle.dt36 = particle.current_reef_ts
                            elif particle.e_num == 37:
                                particle.i37 = particle.current_reef_idx
                                particle.ts37 = particle.current_reef_ts0
                                particle.dt37 = particle.current_reef_ts
                            elif particle.e_num == 38:
                                particle.i38 = particle.current_reef_idx
                                particle.ts38 = particle.current_reef_ts0
                                particle.dt38 = particle.current_reef_ts
                            elif particle.e_num == 39:
                                particle.i39 = particle.current_reef_idx
                                particle.ts39 = particle.current_reef_ts0
                                particle.dt39 = particle.current_reef_ts
                            elif particle.e_num == 40:
                                particle.i40 = particle.current_reef_idx
                                particle.ts40 = particle.current_reef_ts0
                                particle.dt40 = particle.current_reef_ts
                            elif particle.e_num == 41:
                                particle.i41 = particle.current_reef_idx
                                particle.ts41 = particle.current_reef_ts0
                                particle.dt41 = particle.current_reef_ts
                            elif particle.e_num == 42:
                                particle.i42 = particle.current_reef_idx
                                particle.ts42 = particle.current_reef_ts0
                                particle.dt42 = particle.current_reef_ts
                            elif particle.e_num == 43:
                                particle.i43 = particle.current_reef_idx
                                particle.ts43 = particle.current_reef_ts0
                                particle.dt43 = particle.current_reef_ts
                            elif particle.e_num == 44:
                                particle.i44 = particle.current_reef_idx
                                particle.ts44 = particle.current_reef_ts0
                                particle.dt44 = particle.current_reef_ts
                            elif particle.e_num == 45:
                                particle.i45 = particle.current_reef_idx
                                particle.ts45 = particle.current_reef_ts0
                                particle.dt45 = particle.current_reef_ts
                            elif particle.e_num == 46:
                                particle.i46 = particle.current_reef_idx
                                particle.ts46 = particle.current_reef_ts0
                                particle.dt46 = particle.current_reef_ts
                            elif particle.e_num == 47:
                                particle.i47 = particle.current_reef_idx
                                particle.ts47 = particle.current_reef_ts0
                                particle.dt47 = particle.current_reef_ts
                            elif particle.e_num == 48:
                                particle.i48 = particle.current_reef_idx
                                particle.ts48 = particle.current_reef_ts0
                                particle.dt48 = particle.current_reef_ts
                            elif particle.e_num == 49:
                                particle.i49 = particle.current_reef_idx
                                particle.ts49 = particle.current_reef_ts0
                                particle.dt49 = particle.current_reef_ts
                            elif particle.e_num == 50:
                                particle.i50 = particle.current_reef_idx
                                particle.ts50 = particle.current_reef_ts0
                                particle.dt50 = particle.current_reef_ts
                            elif particle.e_num == 51:
                                particle.i51 = particle.current_reef_idx
                                particle.ts51 = particle.current_reef_ts0
                                particle.dt51 = particle.current_reef_ts
                            elif particle.e_num == 52:
                                particle.i52 = particle.current_reef_idx
                                particle.ts52 = particle.current_reef_ts0
                                particle.dt52 = particle.current_reef_ts
                            elif particle.e_num == 53:
                                particle.i53 = particle.current_reef_idx
                                particle.ts53 = particle.current_reef_ts0
                                particle.dt53 = particle.current_reef_ts
                            elif particle.e_num == 54:
                                particle.i54 = particle.current_reef_idx
                                particle.ts54 = particle.current_reef_ts0
                                particle.dt54 = particle.current_reef_ts
                            elif particle.e_num == 55:
                                particle.i55 = particle.current_reef_idx
                                particle.ts55 = particle.current_reef_ts0
                                particle.dt55 = particle.current_reef_ts
                            elif particle.e_num == 56:
                                particle.i56 = particle.current_reef_idx
                                particle.ts56 = particle.current_reef_ts0
                                particle.dt56 = particle.current_reef_ts
                            elif particle.e_num == 57:
                                particle.i57 = particle.current_reef_idx
                                particle.ts57 = particle.current_reef_ts0
                                particle.dt57 = particle.current_reef_ts
                            elif particle.e_num == 58:
                                particle.i58 = particle.current_reef_idx
                                particle.ts58 = particle.current_reef_ts0
                                particle.dt58 = particle.current_reef_ts
                            elif particle.e_num == 59:
                                particle.i59 = particle.current_reef_idx
                                particle.ts59 = particle.current_reef_ts0
                                particle.dt59 = particle.current_reef_ts

                                particle.active = 0 # Deactivate particle, since no more reefs can be saved

                            # Then reset current values to zero
                            particle.current_reef_idx = 0
                            particle.current_reef_ts0 = 0
                            particle.current_reef_ts = 0

                            # Add to event number counter
                            particle.e_num += 1

                        if new_event:
                            # Add status to current (for current event) values
                            # Timesteps at current reef
                            particle.current_reef_ts = 1

                            # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                            particle.current_reef_ts0 = particle.ot - 1

                            # Current reef group
                            particle.current_reef_idx = particle.idx

                    # Finally, check if particle needs to be deleted
                    if particle.ot >= fieldset.max_age:

                        # Only delete particles where at least 1 event has been recorded
                        if particle.e_num > 0:
                            particle.delete()

        return event


    def run(self, **kwargs):
        """
        Run the configured OceanParcels simulation

        """

        if not self.status['particleset']:
            raise Exception('Please run particleset first.')

        self.fh['traj'] = self.dirs['traj'] + self.name + '.nc'

        print('Exporting output to ' + str(self.fh['traj']))

        if self.cfg['test']:
            if self.cfg['test_type'] == 'traj':
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=0.25))
            elif self.cfg['test_type'] == 'vis':
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=0.5))
            else:
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], write_ondelete=True)
        else:
            self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], write_ondelete=True)

        def deleteParticle(particle, fieldset, time):
            #  Recovery kernel to delete a particle if an error occurs
            particle.delete()

        def deactivateParticle(particle, fieldset, time):
            # Recovery kernel to deactivate a particle if an OOB error occurs
            particle.active = 0
            particle.lon = 40
            particle.lat = -1

        # Print some basic statistics
        print('')
        print('Starting simulation:')
        print('Name: ' + self.name)
        print('Number of release cells: ' + str(self.cfg['nsite']) + '/' + str(self.cfg['nsite_nofilter']))
        print('Number of particles released: ' + str(len(self.particles['lon'])))
        print('Release time: ' + str(self.particles['t0'].iloc[0]))
        print('Simulation length: ' + str(self.cfg['run_time']))

        if self.cfg['partitions']:
            print('Partition: ' + str(self.cfg['part']) + '/' + str(self.cfg['partitions']))

        print('')

        # Run the simulation
        self.pset.execute(self.kernel,
                          runtime=self.cfg['run_time'],
                          dt=self.cfg['dt'],
                          recovery={ErrorCode.ErrorOutOfBounds: deactivateParticle,
                                    ErrorCode.ErrorInterpolation: deleteParticle},
                          output_file=self.trajectory_file)

        # Export trajectory file
        self.trajectory_file.export()

        # Add timestep and other details to file
        with Dataset(self.fh['traj'], mode='r+') as nc:
            nc.timestep_seconds = self.cfg['dt'].total_seconds()
            nc.min_competency_seconds = self.cfg['min_competency'].total_seconds()
            nc.max_lifespan_seconds = self.cfg['run_time'].total_seconds()
            nc.larvae_per_cell = self.cfg['pn2']
            nc.total_larvae_released = len(self.particles['lon'])
            nc.interp_method = self.cfg['interp_method']
            nc.e_num = self.cfg['e_num']
            nc.release_year = self.cfg['t0'].year
            nc.release_month = self.cfg['t0'].month
            nc.release_day = self.cfg['t0'].day

            if self.cfg['test']:
                nc.test_mode = 'True'

            nc.partitions = int(self.cfg['partitions'])
            if self.cfg['partitions']:
                nc.part = int(self.cfg['part'])

        self.status['run'] = True


    def generate_dict(self):
        """
        Generate a dict to convert cell indices to reef cover, reef fraction, etc.

        """

        if not self.status['config']:
            raise Exception('Please run config first.')

        if not self.status['fieldset']:
            # Load fields
            self.fh['grid'] = self.dirs['grid'] + self.cfg['grid_filename']
            self.fields = {}

            self.field_list = ['rc', 'rf', 'eez', 'grp', 'idx']

            with Dataset(self.fh['grid'], mode='r') as nc:
                for field in self.field_list:
                    field_varname = self.cfg['grid_' + field + '_varname']
                    self.fields[field] = nc.variables[field_varname][:]

        self.dicts = {}

        # Firstly generate list of indices
        index_list = []
        for (yidx, xidx) in zip(np.ma.nonzero(self.fields['idx'])[0],
                                np.ma.nonzero(self.fields['idx'])[1]):

            index_list.append(self.fields['idx'][yidx, xidx])

        # Now generate dictionaries
        for field in self.field_list:
            if field != 'idx':
                temp_list = []

                for (yidx, xidx) in zip(np.ma.nonzero(self.fields['idx'])[0],
                                        np.ma.nonzero(self.fields['idx'])[1]):

                    temp_list.append(self.fields[field][yidx, xidx])
                    self.dicts[field] = dict(zip(index_list, temp_list))
                    self.dicts[field][0] = -999


        # Create dictionary to translate group -> number of cells in group
        grp_key, grp_val = np.unique(self.fields['grp'].compressed(),return_counts=True)
        self.dicts['grp_numcell'] = dict(zip(grp_key, grp_val))

        self.status['dict'] = True


    def postrun_tests(self):
        """
        This function carries out a test for the accuracy of the events kernels
        or a close-up look of particle trajectories with respect to the model
        grid.

        The kernel function calculates the 'online' settling numbers from the
        test event kernel, and compares them to the postprocessed 'offline'
        numbers.

        """

        if 'test' not in self.cfg.keys():
            raise Exception('Simulation must have been run in testing mode')
        elif not self.cfg['test']:
            raise Exception('Simulation must have been run in testing mode')
        elif not self.status['run']:
            raise Exception('Must run the simulation before events can be tested')
        elif not self.status['dict']:
            self.generate_dict()

        if self.cfg['test_type'] == 'kernel':
            # Convert all units to per year to avoid overflows
            self.cfg['a'] = np.array(self.cfg['test_params']['a']*31536000, dtype=np.float32)
            self.cfg['b'] = np.array(self.cfg['test_params']['b']*31536000, dtype=np.float32)
            self.cfg['tc'] = np.array(self.cfg['test_params']['tc']/31536000, dtype=np.float32)
            self.cfg['??s'] = np.array(self.cfg['test_params']['??s']*31536000, dtype=np.float32)
            self.cfg['??'] = np.array(self.cfg['test_params']['??'], dtype=np.float32)
            self.cfg['??'] = np.array(self.cfg['test_params']['??']*31536000, dtype=np.float32)
            self.cfg['??'] = np.array(self.cfg['test_params']['??'], dtype=np.float32)

            self.cfg['dt']  = np.array(self.cfg['dt'].total_seconds()/31536000, dtype=np.float32)

            with Dataset(self.fh['traj'], mode='r') as nc:
                self.cfg['max_events'] = nc.e_num

                e_num = nc.variables['e_num'][:] # Number of events stored per trajectory
                n_traj = np.shape(e_num)[0] # Number of trajectories in file

                # Load all data into memory
                idx_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.uint16)
                t0_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)
                dt_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)

                fr_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32) # Fraction of reef
                ns_test_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32) # Number/proportion settling from test kernel

                for i in range(self.cfg['max_events']):
                    idx_array[:, i] = nc.variables['i' + str(i)][:, 0]
                    t0_array[:, i] = (nc.variables['ts' + str(i)][:, 0]*self.cfg['dt']-self.cfg['tc']) # Time at arrival
                    dt_array[:, i] = nc.variables['dt' + str(i)][:, 0]*self.cfg['dt'] # Time spent at site
                    ns_test_array[:, i] = nc.variables['Ns' + str(i)][:, 0]

            # Adjust times for events that are partially pre-competent
            idx_array[t0_array + dt_array < 0] = 0 # Corresponds to events that are entirely pre-competent
            dt_array[t0_array < 0] += t0_array[t0_array < 0]
            t0_array[t0_array < 0] = 0

            mask = (idx_array == 0)

            ns_array = np.zeros(np.shape(mask), dtype=np.float32)
            n_traj_reduced = np.shape(mask)[0]
            n_events_reduced = np.shape(mask)[1]

            t0_array[mask] = 0
            dt_array[mask] = 0

            # Now generate an array containing the reef fraction for each index
            def translate(c1, c2):
                # Adapted from Maxim's excellent suggestion:
                # https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
                src, values = np.array(list(c2.keys()), dtype=np.uint16), np.array(list(c2.values()), dtype=np.float32)
                c2_array = np.zeros((src.max()+1), dtype=np.float32)
                c2_array[src] = values
                return c2_array[c1]

            # Now generate an array containing the reef fraction, t0, and dt for each index
            fr_array = translate(idx_array, self.dicts['rf'])
            fr_array[mask] = 0 # Reef fraction
            t0_array[mask] = 0 # Time at arrival
            dt_array[mask] = 0 # Time spent at site

            # Now calculate the fractional losses
            for i in range(n_events_reduced):
                if i == 0:
                    psi0 = np.zeros((n_traj_reduced,), dtype=np.float32)
                    int0 = np.zeros((n_traj_reduced,), dtype=np.float32)
                    t1_prev = np.zeros((n_traj_reduced,), dtype=np.float32)

                fr = fr_array[:, i]
                t0 = t0_array[:, i]
                dt = dt_array[:, i]

                ns_array[:, i], int0 = self.integrate_event(psi0, int0, fr,
                                                            self.cfg['a'],
                                                            self.cfg['b'],
                                                            self.cfg['tc'],
                                                            self.cfg['??s'],
                                                            self.cfg['??'],
                                                            self.cfg['??'],
                                                            self.cfg['??'],
                                                            t0, t1_prev, dt)

                t1_prev = t0 + dt
                psi0 = psi0 + fr*dt

            ns_array = np.ma.masked_array(ns_array, mask=mask)
            ns_test_array = np.ma.masked_array(ns_test_array, mask=mask)

            # Now calculate the percentage difference of events
            pct_diff = 100*(ns_array-ns_test_array)/ns_test_array
            pct_diff = pct_diff.compressed()

            # Plot online-offline difference
            f, ax = plt.subplots(1, 1, figsize=(10, 10))

            xarg_max = np.max(np.abs(pct_diff[np.isfinite(pct_diff)]))
            xarg_max = np.min([xarg_max, 100])
            ax.set_xlim([-xarg_max, xarg_max])
            ax.set_xlabel('Percentage difference between analytical and online settling fluxes')
            ax.set_ylabel('Number of events')
            ax.hist(pct_diff, range=(-xarg_max,xarg_max), bins=200, color='k')

            plt.savefig(self.dirs['fig'] + 'event_accuracy_test.png', dpi=300)

            # Plot larval mortality curves
            f, ax = plt.subplots(1, 1, figsize=(10, 10))

            plt_t0 = 0
            plt_t1 = self.cfg['run_time'].days
            plt_t = np.linspace(plt_t0, plt_t1, num=200)/365

            f_competent = (self.cfg['a']/(self.cfg['a']-self.cfg['b']))*(np.exp(-self.cfg['b']*(plt_t-self.cfg['tc']))-np.exp(-self.cfg['a']*(plt_t-self.cfg['tc'])))
            f_competent[plt_t-self.cfg['tc'] < 0] = 0
            if self.cfg['??'] != 0:
                f_surv = (1 - self.cfg['??']*(self.cfg['??']*(plt_t))**self.cfg['??'])**(1/self.cfg['??'])
            else:
                f_surv = np.exp(-(self.cfg['??']*plt_t)**self.cfg['??'])
            f_comp_surv = f_competent*f_surv

            plt_t[0] = plt_t[1]/10
            ??m = (self.cfg['??']*self.cfg['??']*(self.cfg['??']*plt_t)**(self.cfg['??']-1))/(1-self.cfg['??']*(self.cfg['??']*plt_t)**self.cfg['??'])

            plt_t *= 365

            ax.set_xlim([0, self.cfg['run_time'].days])
            ax.set_ylim([0, 1])
            ax.plot(plt_t/1, f_competent, 'k--', label='Fraction competent')
            ax.plot(plt_t/1, f_surv, 'k:', label='Fraction alive')
            ax.plot(plt_t/1, f_comp_surv, 'k-', label='Fraction alive and competent')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time since release (days)')
            ax.set_ylabel('Proportion of larvae')
            ax.legend(frameon=False, loc=(0.65, 0.9))

            ax2 = ax.twinx()
            ax2.set_yscale('log')
            ax2.plot(plt_t, ??m, 'r-', label='Mortality rate per day')
            ax2.set_ylabel('Mortality rate (1/d)', color='r')
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.tick_right()
            ax2.legend(frameon=False, loc=(0.65,0.85))
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['right'].set_edgecolor('r')
            ax2.tick_params(color='r', labelcolor='r')

            plt.savefig(self.dirs['fig'] + 'mortality_competency_test.png', dpi=300)

            # Print statistics
            pct_within_1pct = np.around(100*np.sum(np.abs(pct_diff) <= 1)/len(pct_diff), decimals=1)
            print(str(pct_within_1pct) + '% of offline settling events within 1% of online value.')

        else:
            # Note that this section may need to be rewritten depending on the
            # minutae of the particular grid being used. Presets for CMEMS and
            # WINDS are supplied here.
            # Load the LSM from the grid file
            with Dataset(self.fh['grid'], mode='r') as nc:
                self.fields['lsm'] = nc.variables[self.cfg['lsm_varname']][:]

            if self.cfg['preset'] == 'CMEMS':
                jmin_psi = np.searchsorted(self.axes['lon_psi'], self.cfg['view_lon0']) - 1
                jmin_psi = 0 if jmin_psi < 0 else jmin_psi
                jmin_rho = jmin_psi
                jmax_psi = np.searchsorted(self.axes['lon_psi'], self.cfg['view_lon1'])
                jmax_rho = jmax_psi + 1

                imin_psi = np.searchsorted(self.axes['lat_psi'], self.cfg['view_lat0']) - 1
                imin_psi = 0 if imin_psi < 0 else imin_psi
                imin_rho = imin_psi
                imax_psi = np.searchsorted(self.axes['lat_psi'], self.cfg['view_lat1'])
                imax_rho = imax_psi + 1

                disp_lon_rho = self.axes['lon_rho'][jmin_rho:jmax_rho]
                disp_lat_rho = self.axes['lat_rho'][imin_rho:imax_rho]
                disp_lon_psi = self.axes['lon_psi'][jmin_psi:jmax_psi]
                disp_lat_psi = self.axes['lat_psi'][imin_psi:imax_psi]

                disp_lsm_psi = self.fields['lsm'][imin_psi:imax_psi, jmin_psi:jmax_psi]

                with Dataset(self.fh['model'][0], mode='r') as nc:
                    # Load the time slice corresponding to release (start)
                    disp_u_rho = nc.variables[self.cfg['u_varname']][0, 0, imin_rho:imax_rho, jmin_rho:jmax_rho]
                    disp_v_rho = nc.variables[self.cfg['v_varname']][0, 0, imin_rho:imax_rho, jmin_rho:jmax_rho]

                # Plot
                f, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.set_xlim(self.cfg['view_lon0'], self.cfg['view_lon1'])
                ax.set_ylim(self.cfg['view_lat0'], self.cfg['view_lat1'])

                # Plot the rho grid
                for i in range(len(disp_lat_rho)):
                    ax.plot([self.cfg['view_lon0'], self.cfg['view_lon1']],
                            [disp_lat_rho[i], disp_lat_rho[i]],
                            'k--', linewidth=0.5)

                for j in range(len(disp_lon_rho)):
                    ax.plot([disp_lon_rho[j], disp_lon_rho[j]],
                            [self.cfg['view_lat0'], self.cfg['view_lat1']],
                            'k--', linewidth=0.5)

                # Plot the lsm_psi mask
                disp_lon_rho_, disp_lat_rho_ = np.meshgrid(disp_lon_rho, disp_lat_rho)
                disp_lon_psi_, disp_lat_psi_ = np.meshgrid(disp_lon_psi, disp_lat_psi)

                ax.pcolormesh(disp_lon_rho, disp_lat_rho, disp_lsm_psi, cmap=cmr.copper,
                              vmin=-0.5, vmax=1.5)

                # Plot the velocity field
                ax.quiver(disp_lon_rho, disp_lat_rho, disp_u_rho, disp_v_rho)

                # Load the trajectories
                with Dataset(self.fh['traj'], mode='r') as nc:
                    plat = nc.variables['lat'][:]
                    plon = nc.variables['lon'][:]

                for particle in range(np.shape(plat)[0]):
                    ax.plot(plon[particle, :], plat[particle, :], 'w-', linewidth=0.5)

                plt.savefig(self.dirs['fig'] + 'trajectory_test.png', dpi=300)

            elif self.cfg['preset'] == 'WINDS':
                jmin_psi = np.searchsorted(self.axes['lon_psi'], self.cfg['view_lon0']) - 1
                jmin_psi = 0 if jmin_psi < 0 else jmin_psi
                jmin_rho = jmin_psi
                jmax_psi = np.searchsorted(self.axes['lon_psi'], self.cfg['view_lon1']) + 1
                jmax_rho = jmax_psi + 1

                imin_psi = np.searchsorted(self.axes['lat_psi'], self.cfg['view_lat0']) - 1
                imin_psi = 0 if imin_psi < 0 else imin_psi
                imin_rho = imin_psi
                imax_psi = np.searchsorted(self.axes['lat_psi'], self.cfg['view_lat1']) + 1
                imax_rho = imax_psi + 1

                disp_lon_rho = self.axes['lon_rho'][jmin_rho:jmax_rho]
                disp_lat_rho = self.axes['lat_rho'][imin_rho:imax_rho]
                disp_lon_psi = self.axes['lon_psi'][jmin_psi:jmax_psi]
                disp_lat_psi = self.axes['lat_psi'][imin_psi:imax_psi]

                disp_lsm_rho = self.fields['lsm'][imin_rho:imax_rho, jmin_rho:jmax_rho]

                with Dataset(self.fh['model'][0], mode='r') as nc:
                    # Load the time slice corresponding to release (start)
                    disp_u = nc.variables[self.cfg['u_varname']][0, imin_rho:imax_rho, jmin_psi:jmax_psi]
                    disp_v = nc.variables[self.cfg['v_varname']][0, imin_psi:imax_psi, jmin_rho:jmax_rho]

                    disp_u_lon = nc.variables[self.cfg['lon_u_dimname']][imin_rho:imax_rho, jmin_psi:jmax_psi]
                    disp_u_lat = nc.variables[self.cfg['lat_u_dimname']][imin_rho:imax_rho, jmin_psi:jmax_psi]
                    disp_v_lon = nc.variables[self.cfg['lon_v_dimname']][imin_psi:imax_psi, jmin_rho:jmax_rho]
                    disp_v_lat = nc.variables[self.cfg['lat_v_dimname']][imin_psi:imax_psi, jmin_rho:jmax_rho]

                # Plot
                f, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.set_xlim(self.cfg['view_lon0'], self.cfg['view_lon1'])
                ax.set_ylim(self.cfg['view_lat0'], self.cfg['view_lat1'])

                # Plot the psi grid
                for i in range(len(disp_lat_psi)):
                    ax.plot([self.cfg['view_lon0'], self.cfg['view_lon1']],
                            [disp_lat_psi[i], disp_lat_psi[i]],
                            'k--', linewidth=0.5)

                for j in range(len(disp_lon_psi)):
                    ax.plot([disp_lon_psi[j], disp_lon_psi[j]],
                            [self.cfg['view_lat0'], self.cfg['view_lat1']],
                            'k--', linewidth=0.5)

                # Plot the rho mask
                disp_lon_rho_, disp_lat_rho_ = np.meshgrid(disp_lon_rho, disp_lat_rho)
                disp_lon_psi_, disp_lat_psi_ = np.meshgrid(disp_lon_psi, disp_lat_psi)

                ax.pcolormesh(disp_lon_psi, disp_lat_psi, disp_lsm_rho[1:-1, 1:-1], cmap=cmr.copper,
                              vmin=-0.5, vmax=1.5)

                # Plot the velocity field
                ax.quiver(disp_u_lon, disp_u_lat, disp_u, np.zeros_like(disp_u), scale=10)
                ax.quiver(disp_v_lon, disp_v_lat, np.zeros_like(disp_v), disp_v, scale=10)

                # Load the trajectories
                with Dataset(self.fh['traj'], mode='r') as nc:
                    plat = nc.variables['lat'][:]
                    plon = nc.variables['lon'][:]

                for particle in range(np.shape(plat)[0]):
                    ax.plot(plon[particle, :], plat[particle, :], 'w-', linewidth=0.5)

                plt.savefig(self.dirs['fig'] + 'trajectory_test.png', dpi=300)


    def generate_matrix(self, **kwargs):

        """
        Parameters (* are required)
        ----------
        kwargs :
            fh*: File handles to data
            parameters*: Postproc parameters (in dict)
            filters: Dict of filters for eez/grp
            subset: Integer factor to subset particle IDs by
            e_num_ceil: Maximum number of events to consider
            numexpr: Whether to use numexpr acceleration


        Important note: all units are converted to DAYS within this function

        """

        # Define conversion factor (probably seconds per year)
        conv_f = 31536000

        # Cumulative day total for a year
        day_mo = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        day_mo_cs = np.cumsum(day_mo)
        day_mo_ly = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        day_mo_ly_cs = np.cumsum(day_mo_ly)

        if not self.status['dict']:
            self.generate_dict()

        if 'parameters' not in kwargs.keys():
            raise KeyError('Please supply a parameters dictionary.')
        else:
            # Convert all units to years to prevent overflows from large numbers
            self.cfg['a'] = np.array(kwargs['parameters']['a']*conv_f, dtype=np.float32)
            self.cfg['b'] = np.array(kwargs['parameters']['b']*conv_f, dtype=np.float32)
            self.cfg['tc'] = np.array(kwargs['parameters']['tc']/conv_f, dtype=np.float32)
            self.cfg['??s'] = np.array(kwargs['parameters']['??s']*conv_f, dtype=np.float32)
            self.cfg['??'] = np.array(kwargs['parameters']['??'], dtype=np.float32)
            self.cfg['??'] = np.array(kwargs['parameters']['??']*conv_f, dtype=np.float32)
            self.cfg['??'] = np.array(kwargs['parameters']['??'], dtype=np.float32)

        if 'fh' not in kwargs.keys():
            raise KeyError('Please supply a list of files to analyse.')

        if 'rpm' not in self.cfg.keys():
            raise KeyError('Please supply the number of separate releases per month in config.')

        self.cfg['ldens'] = 1 # There's no good reason to change this yet

        if 'subset' in kwargs.keys():
            self.cfg['subset'] = int(kwargs['subset'])
        else:
            self.cfg['subset'] = False

        if 'numexpr' in kwargs.keys():
            self.integrate = self.integrate_event_numexpr if kwargs['numexpr'] else self.integrate_event
        else:
            self.integrate = self.integrate_event_numexpr

        # Define translation function
        def translate(c1, c2):
            # Adapted from Maxim's excellent suggestion:
            # https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
            src, values = np.array(list(c2.keys()), dtype=np.uint16), np.array(list(c2.values()), dtype=np.float32)
            c2_array = np.zeros((src.max()+1), dtype=np.float32)
            c2_array[src] = values
            return c2_array[c1]

        # Get files
        fh_list = sorted(glob(self.dirs['traj'] + kwargs['fh']))

        # Open the first file to find the number of events stored and remaining parameters
        with xr.open_zarr(fh_list[0]) as file:
            self.cfg['max_events'] = file.attrs['e_num']

            self.cfg['dt'] = int(file.attrs['timestep_seconds'])/conv_f
            self.cfg['lpc'] = int(file.attrs['larvae_per_cell'])

            if self.cfg['subset']:
                if self.cfg['lpc']/self.cfg['subset'] < 16:
                    raise Exception('Subset too high - aliasing may be an issue.')
                else:
                    self.cfg['lpc'] /= self.cfg['subset']

            if self.cfg['tc']*conv_f < int(file.attrs['min_competency_seconds']):
                print('Warning: minimum competency chosen is smaller than the value used at run-time (' + str(int(file.attrs['min_competency_seconds'])) +'s).')

        # Get the full time range
        t0_list = []

        for fh in fh_list:
            with xr.open_zarr(fh) as file:
                y0 = int(file.attrs['release_year'])
                t0_list.append(y0)

        t0_list = np.array(t0_list)

        # Check all files are present
        if len(np.unique(t0_list)) != 1:
            raise Exception('More than year present!')
        else:
            y0 = t0_list[0]

        ly = False if y0%4 else True
        n_days = 366 if ly else 365

        if n_days != len(fh_list):
            raise Exception('Warning: there is an unexpected number of files!')

        # Get a list of group IDs
        source_reef_mask = (self.fields['rc'] > 0)
        sink_reef_mask = (self.fields['rc'] > 0)

        if 'source_filters' in kwargs:
            for filter_name in kwargs['source_filters'].keys():
                source_reef_mask *= np.isin(self.fields[filter_name], kwargs['source_filters'][filter_name])

        if 'sink_filters' in kwargs:
            for filter_name in kwargs['sink_filters'].keys():
                sink_reef_mask *= np.isin(self.fields[filter_name], kwargs['sink_filters'][filter_name])

        source_grp_list = np.unique(np.ma.masked_where(source_reef_mask == 0, self.fields['grp'])).compressed()
        source_grp_bnds = np.append(source_grp_list, source_grp_list[-1]+1)-0.5
        source_grp_num = len(source_grp_list)

        sink_grp_list = np.unique(np.ma.masked_where(sink_reef_mask == 0, self.fields['grp'])).compressed()
        sink_grp_bnds = np.append(sink_grp_list, sink_grp_list[-1]+1)-0.5
        sink_grp_num = len(sink_grp_list)

        # Set up matrices
        matrix1 = np.zeros((source_grp_num, sink_grp_num, n_days), dtype=np.float32) # For number settling
        matrix2 = np.zeros_like(matrix1, dtype=np.float32) # For number settling * reef fraction
        matrix3 = np.zeros_like(matrix1, dtype=np.float32) # For number settling * reef fraction * transit time
        matrix4 = np.zeros_like(matrix1, dtype=np.int32) # For number of events

        # Create attribute dictionary
        attr_dict = {}

        for fhi, fh in tqdm(enumerate(fh_list), total=len(fh_list)):
            with xr.open_zarr(fh, mask_and_scale=False) as file:
                n_traj = np.shape(file.variables['e_num'][:])[0] # Number of trajectories in file/subset

                if not n_traj:
                    # Skip if there are no trajectories stored in file
                    raise Exception('No trajectories found in file ' + str(fh) + '!')

                # Extract origin date from filename
                y0 = int(file.attrs['release_year'])
                m0 = int(file.attrs['release_month'])
                d0 = int(file.attrs['release_day'])
                assert y0 < 2025 and y0 > 1990
                assert m0 < 13 and m0 > 0
                assert d0 < 32 and d0 > 0
                t0 = datetime(year=y0, month=m0, day=d0, hour=0)

                # Load all data into memory
                idx_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.uint16)
                t0_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)
                dt_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)
                ns_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)

                for i in range(self.cfg['max_events']):
                    idx_array[:, i] = file['i' + str(i)].values
                    t0_array[:, i] = file['ts' + str(i)].values*self.cfg['dt']-self.cfg['tc'] # Time at arrival
                    dt_array[:, i] = file['dt' + str(i)].values*self.cfg['dt'] # Time at site

                idx0_array = file['idx0'].values

                # Adjust times for events that are partially pre-competent
                idx_array[t0_array + dt_array < 0] = 0 # Corresponds to events that are entirely pre-competent
                dt_array[t0_array < 0] += t0_array[t0_array < 0]
                t0_array[t0_array < 0] = 0

                mask = (idx_array == 0)

                ns_array = np.zeros(np.shape(mask), dtype=np.float32)
                n_traj_reduced = np.shape(mask)[0]
                n_events_reduced = np.shape(mask)[1]

                # Copy over parameters for output file
                for attr_name in ['parcels_version', 'timestep_seconds',
                                  'min_competency_seconds', 'max_lifespan_seconds',
                                  'larvae_per_cell', 'interp_method', 'e_num',
                                  'release_year']:
                    if fhi == 0:
                        attr_dict[attr_name] = file.attrs[attr_name]
                    else:
                        assert attr_dict[attr_name] == file.attrs[attr_name]

                if fhi == 0:
                    attr_dict['total_larvae_released'] = file.attrs['total_larvae_released']
                    self.cfg['run_time'] = timedelta(seconds=int(attr_dict['max_lifespan_seconds']))
                else:
                    attr_dict['total_larvae_released'] += file.attrs['total_larvae_released']

            # Now generate an array containing the reef fraction, t0, and dt for each index
            fr_array = translate(idx_array, self.dicts['rf'])

            # Set fr/t0/dt to 0 for all invalid events (to prevent pre-competent events
            # from ruining integrals)
            fr_array[mask] = 0 # Reef fraction
            t0_array[mask] = 0 # Time at arrival
            dt_array[mask] = 0 # Time spent at site

            for i in range(n_events_reduced):
                if i == 0:
                    psi0 = np.zeros((n_traj_reduced,), dtype=np.float32)
                    int0 = np.zeros((n_traj_reduced,), dtype=np.float32)
                    t1_prev = np.zeros((n_traj_reduced,), dtype=np.float32)

                fr = fr_array[:, i]
                t0 = t0_array[:, i]
                dt = dt_array[:, i]

                ns_array[:, i], int0 = self.integrate(psi0, int0, fr,
                                                      self.cfg['a'],
                                                      self.cfg['b'],
                                                      self.cfg['tc'],
                                                      self.cfg['??s'],
                                                      self.cfg['??'],
                                                      self.cfg['??'],
                                                      self.cfg['??'],
                                                      t0, t1_prev, dt)

                t1_prev = t0 + dt
                psi0 = psi0 + fr*dt

            # Compress earlier to accelerate calculations
            ns_array = np.ma.masked_array(ns_array, mask=mask).compressed()
            t0_array = np.ma.masked_array(t0_array, mask=mask).compressed()
            dt_array = np.ma.masked_array(dt_array, mask=mask).compressed()

            assert np.all(ns_array >= 0)
            assert np.all(t0_array >= 0)
            assert np.all(dt_array >= 0)

            # From the index array, extract group
            grp_array = np.floor(idx_array/(2**8)).astype(np.uint8)
            grp_array = np.ma.masked_array(grp_array, mask=mask).compressed()

            # Extract origin group and project
            grp0 = np.floor(idx0_array/(2**8)).astype(np.uint8)
            grp0_array = np.zeros_like(idx_array, dtype=np.uint8)
            grp0_array[:] = grp0.reshape((-1, 1))
            grp0_array = np.ma.masked_array(grp0_array, mask=mask).compressed()

            # Extract origin reef cover and project
            rc0_array = np.zeros_like(idx_array, dtype=np.float32)
            rc0_array[:] = translate(idx0_array, self.dicts['rc']).reshape((-1, 1))
            rc0_array = np.ma.masked_array(rc0_array, mask=mask).compressed()

            filter_mask = 1-np.isin(grp0_array, source_grp_list)*np.isin(grp0_array, sink_grp_list)
            grp_i_array = np.ma.masked_array(grp0_array, mask=filter_mask).compressed()
            grp_j_array = np.ma.masked_array(grp_array, mask=filter_mask).compressed()
            rc_i_array = np.ma.masked_array(rc0_array, mask=filter_mask).compressed()
            ns_ij_array = np.ma.masked_array(ns_array, mask=filter_mask).compressed()
            t0_ij_array = np.ma.masked_array(t0_array, mask=filter_mask).compressed()
            dt_j_array = np.ma.masked_array(dt_array, mask=filter_mask).compressed()

            # Find time index
            if ly:
                ti = day_mo_ly_cs[m0-1] + d0 - 1
            else:
                ti = day_mo_cs[m0-1] + d0 - 1

            # Now grid quantities:
            # p(i, j, t) = sum(ns[i, j])/(lpc[i]*rpm*cpg[i])
            #              ns: fraction of released larvae from i settling at j
            #              lpc: larvae per cell
            #              rpm: releases per month
            #              cpg: cells per release group

            # f(i, j, t) = sum(ns[i, j]*rc[i]*ldens)
            #              rc: reef cover (m2)
            #              ldens: larvae per unit reef cover (m2) per month

            # t(i, j, t) = sum(f(i, j, t)*(t0[i, j]+0.5*dt[i, j]))/sum(f(i, j, t))
            #              Note that this is the flux-weighted time mean
            #              t0 + dt: time taken for larva to travel from i to j

            # Therefore (for rpm = 1, lpc = const):
            # p[i1+i2, j1+j2, t] = (ns[i1, j1, t] + ns[i2, j2, t])/(lpc*(cpg[i1]+cpg[i2]))
            # f[i1+i2, j1+j2, t] = ldens*(ns[i1, j1, t]*rc[i1] + ns[i2, j2, t]*rc[i2])
            # t[i1+i2, j1+j2, t] = (ns[i1, j1, t]*rc[i1]*(t0[i1, j1]+0.5*dt[i1, j1]) + ns[i2, j2, t]*rc[i2]*(t0[i2, j2]+0.5*dt[i2, j2]))/(ns[i1, j1, t]*rc[i1] + ns[i2, j2, t]*rc[i2])

            # So we need to save the following quantities:
            # ns[i, j, t]
            # ns[i, j, t]*rc[i]
            # ns[i, j, t]*rc[i]*Dt[i, j]
            # cpg[i]

            matrix1[:, :, ti] += np.histogram2d(grp_i_array, grp_j_array,
                                                bins=[source_grp_bnds, sink_grp_bnds],
                                                weights=ns_ij_array)[0]

            matrix2[:, :, ti] += np.histogram2d(grp_i_array, grp_j_array,
                                                bins=[source_grp_bnds, sink_grp_bnds],
                                                weights=ns_ij_array*rc_i_array*self.cfg['ldens'])[0]

            matrix3[:, :, ti] += np.histogram2d(grp_i_array, grp_j_array,
                                                bins=[source_grp_bnds, sink_grp_bnds],
                                                weights=(ns_ij_array*rc_i_array*self.cfg['ldens']*
                                                         (self.cfg['tc'] + t0_ij_array + 0.5*dt_j_array)))[0]

            matrix4[:, :, ti] += np.histogram2d(grp_i_array, grp_j_array,
                                                bins=[source_grp_bnds, sink_grp_bnds])[0].astype(np.int32)

        # Now convert to xarray
        matrix = xr.Dataset(data_vars=dict(ns=(['source_group', 'sink_group', 'time'], matrix1,
                                               {'full_name': 'normalised_settling_larvae',
                                                'weights': 'ns_ij'}),
                                           ns_rc=(['source_group', 'sink_group', 'time'], matrix2,
                                                  {'full_name': 'rf-weighted_settling_larvae',
                                                   'weights': 'ns_ij*rf_i'}),
                                           ns_rc_t=(['source_group', 'sink_group', 'time'], matrix3,
                                                    {'full_name': 'rf-t-weighted_settling_larvae',
                                                     'weights': 'ns_ij*t_ij*rf_i'}),
                                           en=(['source_group', 'sink_group', 'time'], matrix4,
                                               {'full_name': 'number_of_events_registered',
                                                'weights': '1'}),
                                           cpg=(['source_group'], translate(source_grp_list, self.dicts['grp_numcell']),
                                                {'full_name': 'cells_per_group'})),
                            coords=dict(source_group=source_grp_list, sink_group=sink_grp_list,
                                        time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                           periods=n_days, freq='D')),
                            attrs=dict(a=self.cfg['a'],
                                       b=self.cfg['b'],
                                       tc=self.cfg['tc'],
                                       ??s=self.cfg['??s'],
                                       ??=self.cfg['??'],
                                       ??=self.cfg['??'],
                                       ??=self.cfg['??'],
                                       configuration=self.cfg['preset'],
                                       parcels_version=attr_dict['parcels_version'],
                                       timestep_seconds=attr_dict['timestep_seconds'],
                                       min_competency_seconds=attr_dict['min_competency_seconds'],
                                       max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                       larvae_per_cell=attr_dict['larvae_per_cell'],
                                       interp_method=attr_dict['interp_method'],
                                       e_num=attr_dict['e_num'],
                                       total_larvae_released=attr_dict['total_larvae_released'],))

        self.status['matrix'] = True

        return matrix

    def plot_parameters(self, **kwargs):

        """
        Parameters (* are required)
        ----------
        kwargs :
            fh*: File handle for figure
            parameters*: Postproc parameters (in dict) [if matrix not yet run]

        """

        if not self.status['matrix']:
            if 'parameters' not in kwargs:
                raise Exception('Please supply biological parameters or generate matrix first.')
            else:
                # Convert all units to days to prevent overflows from large numbers
                self.cfg['a'] = np.array(kwargs['parameters']['a']*31536000, dtype=np.float64)
                self.cfg['b'] = np.array(kwargs['parameters']['b']*31536000, dtype=np.float64)
                self.cfg['tc'] = np.array(kwargs['parameters']['tc']/31536000, dtype=np.float64)
                self.cfg['??s'] = np.array(kwargs['parameters']['??s']*31536000, dtype=np.float64)
                self.cfg['??'] = np.array(kwargs['parameters']['??'], dtype=np.float64)
                self.cfg['??'] = np.array(kwargs['parameters']['??']*31536000, dtype=np.float64)
                self.cfg['??'] = np.array(kwargs['parameters']['??'], dtype=np.float64)

        if 'fh' not in kwargs:
            print('Using default file name.')
            fh = 'biological_parameters.png'
        else:
            fh = kwargs['fh']

        # Plot larval mortality curves
        f, ax = plt.subplots(1, 1, figsize=(10, 10))

        plt_t0 = 0
        plt_t1 = 120
        plt_t = np.linspace(plt_t0, plt_t1, num=200)/365
        plt_t[0] = plt_t[1]/2 # Plotting hack to prevent division by zero

        f_competent = (self.cfg['a']/(self.cfg['a']-self.cfg['b']))*(np.exp(-self.cfg['b']*(plt_t-self.cfg['tc']))-np.exp(-self.cfg['a']*(plt_t-self.cfg['tc'])))
        f_competent[plt_t-self.cfg['tc'] < 0] = 0

        if self.cfg['??'] != 0:
            f_surv = (1 - self.cfg['??']*(self.cfg['??']*(plt_t))**self.cfg['??'])**(1/self.cfg['??'])
            ??m = (self.cfg['??']*self.cfg['??']*(self.cfg['??']*plt_t)**(self.cfg['??']-1))/(1-self.cfg['??']*(self.cfg['??']*plt_t)**self.cfg['??'])
        else:
            f_surv = np.exp(-(self.cfg['??']*plt_t)**self.cfg['??'])
            ??m = (self.cfg['??']*self.cfg['??'])*(self.cfg['??']*plt_t)**(self.cfg['??']-1)

        f_comp_surv = f_competent*f_surv

        plt_t[0] = plt_t[1]/10
        plt_t *= 365

        ax.set_xlim([0, self.cfg['run_time'].days])
        ax.set_ylim([0, 1])
        ax.plot(plt_t/1, f_competent, 'k--', label='Fraction competent')
        ax.plot(plt_t/1, f_surv, 'k:', label='Fraction alive')
        ax.plot(plt_t/1, f_comp_surv, 'k-', label='Fraction alive and competent')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time since release (days)')
        ax.set_ylabel('Proportion of larvae')
        ax.legend(frameon=False, loc=(0.65, 0.9))

        ax2 = ax.twinx()
        ax2.set_yscale('log')
        ax2.plot(plt_t, ??m, 'r-', label='Mortality rate per day')
        ax2.set_ylabel('Mortality rate (1/d)', color='r')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax2.legend(frameon=False, loc=(0.65,0.85))
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['right'].set_edgecolor('r')
        ax2.tick_params(color='r', labelcolor='r')

        plt.savefig(self.dirs['fig'] + fh, dpi=300)
