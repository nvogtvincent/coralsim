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
import matplotlib.colors as colors
import matplotlib
import cmasher as cmr
import pandas as pd
import cartopy.crs as ccrs
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
        config: register directories and load preset
        generate_fieldset: generate fieldsets for OceanParcels
        generate_particleset: generate initial conditions for particles + kernels
        run: run OceanParcels using the above configuration
    """


    def __init__(self):
        # Set up a status dictionary so we know the completion status of the
        # experiment configuration

        self.status = {'config': False,
                       'fieldset': False,
                       'particleset': False,
                       'run': False}

        # Set up dictionaries for various parameters
        # self.
        # self.fh = {}
        # self.params = {}

        # self.root_dir = root_dir


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
                 'grid_cc_varname' : 'coral_cover_c', # Coral cover
                 'grid_cf_varname' : 'coral_frac_c',  # Coral fraction
                 'grid_eez_varname': 'coral_eez_c',   # Coral EEZ
                 'grid_grp_varname': 'coral_grp_c',   # Coral group
                 'grid_idx_varname': 'coral_idx_c',   # Coral index,

                 # Variable types
                 'cc_dtype': np.int32,
                 'cf_dtype': np.float32,
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
                 'grid' : 'A'
                 }

        PRESETS = {'CMEMS': CMEMS}

        if kwargs['preset'] not in PRESETS.keys():
            raise KeyError('Preset not recognised.')

        self.cfg = PRESETS[kwargs['preset']]
        self.dirs = dir_dict
        self.fh = {}

        self.status['config'] = True


    def generate_fieldset(self, **kwargs):
        """
        Generate the FieldSet object for OceanParcels

        Parameters
        ----------
        **kwargs : interp_method = 'freeslip' or 'linear'

        """

        if not self.status['config']:
            raise Exception('Please run config first.')

        # Generate file names
        self.fh['grid'] = self.dirs['grid'] + self.cfg['grid_filename']
        self.fh['model'] = sorted(glob(self.dirs['model'] + self.cfg['model_filenames']))

        # Import grid axes
        self.axes = {}

        if self.cfg['grid'] == 'A':
            with Dataset(self.fh['grid'], mode='r') as nc:
                self.axes['lon_rho'] = np.array(nc.variables[self.cfg['lon_rho_dimname']][:])
                self.axes['lat_rho'] = np.array(nc.variables[self.cfg['lat_rho_dimname']][:])
                self.axes['nx_rho'] = len(self.axes['lon_rho'])
                self.axes['ny_rho'] = len(self.axes['lat_rho'])

                self.axes['lon_psi'] = np.array(nc.variables[self.cfg['lon_psi_dimname']][:])
                self.axes['lat_psi'] = np.array(nc.variables[self.cfg['lat_psi_dimname']][:])
                self.axes['nx_psi'] = len(self.axes['lon_psi'])
                self.axes['ny_psi'] = len(self.axes['lat_psi'])
        elif self.cfg['grid'] == 'C':
            raise NotImplementedError('C-grids have not yet been implemented.')
        else:
            raise KeyError('Grid type not understood.')

        # Import currents
        if 'interp_method' not in kwargs.keys():
            print('No interpolation method prescribed for velocities.')
            print('Setting velocity interpolation method to linear.')
            print('')
            self.cfg['interp_method'] = 'linear'
        elif kwargs['interp_method'] not in ['linear', 'cgrid_velocity', 'freeslip', 'partialslip']:
            raise KeyError('Velocity interpolation method not understood.')
        else:
            self.cfg['interp_method'] = kwargs['interp_method']

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
        else:
            raise NotImplementedError('C-grids have not yet been implemented.')

        self.fields = {}

        # Import additional fields
        if self.cfg['grid'] == 'A':
            field_list = ['cc', 'cf', 'eez', 'grp', 'idx']

            for field in field_list:
                field_varname = self.cfg['grid_' + field + '_varname']

                # Firstly verify that dimensions are correct
                with Dataset(self.fh['grid'], mode='r') as nc:
                    self.fields[field] = nc.variables[field_varname][:]

                if not np.array_equiv(np.shape(self.fields[field]),
                                      (self.axes['ny_psi'], self.axes['nx_psi'])):
                    raise Exception('Field ' + field_varname + ' has incorrect dimensions')

                if field in ['cc', 'eez', 'grp', 'idx']:
                    if np.max(self.fields[field]) > np.iinfo(self.cfg[field + '_dtype']).max:
                        raise Exception('Maximum value exceeded in ' + field_varname + '.')

                # Use OceanParcels routine to import field
                scratch_field = Field.from_netcdf(self.fh['grid'],
                                                  variable=self.cfg['grid_' + str(field) + '_varname'],
                                                  dimensions={'lon': self.cfg['lon_psi_dimname'],
                                                              'lat': self.cfg['lat_psi_dimname']},
                                                  interp_method='nearest', mesh='spherical',
                                                  allow_time_extrapolation=True)

                self.fieldset.add_field(scratch_field)

        else:
            raise NotImplementedError('C-grids have not yet been implemented.')

        self.status['fieldset'] = True


    def generate_particleset(self, **kwargs):

        """
        Generate the ParticleSet object for OceanParcels

        Parameters
        ----------
        **kwargs : num = Number of particles to (aim to) release per cell

                   t0 = Release time for particles (datetime)
                   competency = Competency period (timedelta)
                   dt = Model time-step (timedelta)
                   run_time = Model run-time (timedelta)

                   test = Whether to activate testing kernels (bool)

                   filters = Dict with 'eez' and/or 'grp' keys to enable filter
                             for release sites
                   plot = 'grp' or 'eez' or None - plots the specified field if
                          not None
                   plot_fh = File handle to save plot under in figures directory

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

        if 'filters' not in kwargs.keys():
            self.cfg['filters'] = None
        else:
            for filter_name in kwargs['filters'].keys():
                if filter_name not in ['eez', 'grp']:
                    raise KeyError('Filter name ' + filter_name + ' not understood.')

        if 'competency' in kwargs.keys():
            self.cfg['competency'] = kwargs['competency']
        else:
            print('Minimum competency period not supplied.')
            print('Setting to default of 5 days.')
            print('')
            self.cfg['competency'] = timedelta(days=5)

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
            if kwargs['test'] in ['kernel', 'traj', False]:
                if kwargs['test'] in ['kernel', 'traj']:
                    self.cfg['test'] = True
                    self.cfg['test_type'] = kwargs['test']
                else:
                    self.cfg['test'] = False
            else:
                print('Test type not understood. Ignoring test.')
                self.cfg['test'] = False
        else:
            self.cfg['test'] = False

        if 'plot' in kwargs.keys():
            self.cfg['plot'] = kwargs['plot']

            if kwargs['plot'] not in ['grp', 'eez']:
                raise KeyError('Plot type not understood. Setting to default of EEZ.')
                self.cfg['plot_type'] = 'grp'
            else:
                self.cfg['plot'] = True
                self.cfg['plot_type'] = kwargs['plot']

            if 'plot_fh' not in kwargs.keys():
                raise KeyError('Plot type not understood. Setting to \'particle.png\'.')
                self.fh['fig'] = self.dirs['fig'] + 'particle.png'
            else:
                self.fh['fig'] = self.dirs['fig'] + kwargs['plot_fh']

        else:
            self.cfg['plot'] = False

        # Build a mask of valid initial position cells
        reef_mask = (self.fields['cc'] > 0)
        self.cfg['nsite_nofilter'] = int(np.sum(reef_mask))

        # Filter if applicable
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

        # Now initialise particle properties
        field_list = ['cc', 'cf', 'eez', 'grp', 'idx']

        for field in field_list:
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

                for field in field_list:
                    value_k = self.fields[field][i, j]
                    particles[field][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = value_k

        else:
            raise NotImplementedError('C-grids have not been implemented yet!')

        # Now export to DataFrame
        particles_df = pd.DataFrame({'lon': particles['lon'],
                                     'lat': particles['lat']})

        for field in field_list:
            particles_df[field] = particles[field]

        # Now add release times
        particles_df['t0'] = self.cfg['t0']

        # Save particles_df to class
        self.particles = particles_df

        # Set up the particle class
        self.larva = self.build_larva(self.cfg['test'])

        # Override for the trajectory testing mode
        if self.cfg['test']:
            if self.cfg['test_type'] == 'traj':
                # Override the run time (no long run time is needed for
                # these experiments) and set t0 to first time frame
                self.cfg['t0'] = model_start
                self.cfg['run_time'] = timedelta(days=20)
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

        # Add competency period and maximum age to fieldset
        self.fieldset.add_constant('competency', int(self.cfg['competency']/self.cfg['dt']))
        self.fieldset.add_constant('max_age', int(self.cfg['run_time']/self.cfg['dt']))
        assert self.fieldset.max_age < np.iinfo(np.uint16).max

        # Generate kernels
        self.kernel = (self.pset.Kernel(AdvectionRK4) + self.pset.Kernel(self.build_event_kernel(self.cfg['test'])))

        if self.cfg['test']:
            self.cfg['test_parameters'] = {'lm': 8e-7,
                                           'ls': 1e-5}

        # Now plot (if wished)
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

            plt.savefig(self.fh['fig'], dpi=300)
            plt.close()

        self.status['particleset'] = True


    def build_larva(self, test):
        """
        This script builds the larva class as a test or operational class based
        on whether test is True or False

        """

        if type(test) != bool:
            raise Exception('Input must be a boolean.')

        if test:
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
                i0 = Variable('i0', dtype=np.int32, initial=0, to_write=True)
                ts0 = Variable('ts0', dtype=np.int16, initial=0, to_write=True)
                dt0 = Variable('dt0', dtype=np.int16, initial=0, to_write=True)

                i1 = Variable('i1', dtype=np.int32, initial=0, to_write=True)
                ts1 = Variable('ts1', dtype=np.int16, initial=0, to_write=True)
                dt1 = Variable('dt1', dtype=np.int16, initial=0, to_write=True)

                i2 = Variable('i2', dtype=np.int32, initial=0, to_write=True)
                ts2 = Variable('ts2', dtype=np.int16, initial=0, to_write=True)
                dt2 = Variable('dt2', dtype=np.int16, initial=0, to_write=True)

                i3 = Variable('i3', dtype=np.int32, initial=0, to_write=True)
                ts3 = Variable('ts3', dtype=np.int16, initial=0, to_write=True)
                dt3 = Variable('dt3', dtype=np.int16, initial=0, to_write=True)

                i4 = Variable('i4', dtype=np.int32, initial=0, to_write=True)
                ts4 = Variable('ts4', dtype=np.int16, initial=0, to_write=True)
                dt4 = Variable('dt4', dtype=np.int16, initial=0, to_write=True)

                i5 = Variable('i5', dtype=np.int32, initial=0, to_write=True)
                ts5 = Variable('ts5', dtype=np.int16, initial=0, to_write=True)
                dt5 = Variable('dt5', dtype=np.int16, initial=0, to_write=True)

                i6 = Variable('i6', dtype=np.int32, initial=0, to_write=True)
                ts6 = Variable('ts6', dtype=np.int16, initial=0, to_write=True)
                dt6 = Variable('dt6', dtype=np.int16, initial=0, to_write=True)

                i7 = Variable('i7', dtype=np.int32, initial=0, to_write=True)
                ts7 = Variable('ts7', dtype=np.int16, initial=0, to_write=True)
                dt7 = Variable('dt7', dtype=np.int16, initial=0, to_write=True)

                i8 = Variable('i8', dtype=np.int32, initial=0, to_write=True)
                ts8 = Variable('ts8', dtype=np.int16, initial=0, to_write=True)
                dt8 = Variable('dt8', dtype=np.int16, initial=0, to_write=True)

                i9 = Variable('i9', dtype=np.int32, initial=0, to_write=True)
                ts9 = Variable('ts9', dtype=np.int16, initial=0, to_write=True)
                dt9 = Variable('dt9', dtype=np.int16, initial=0, to_write=True)

                i10 = Variable('i10', dtype=np.int32, initial=0, to_write=True)
                ts10 = Variable('ts10', dtype=np.int16, initial=0, to_write=True)
                dt10 = Variable('dt10', dtype=np.int16, initial=0, to_write=True)

                i11 = Variable('i11', dtype=np.int32, initial=0, to_write=True)
                ts11 = Variable('ts11', dtype=np.int16, initial=0, to_write=True)
                dt11 = Variable('dt11', dtype=np.int16, initial=0, to_write=True)

                i12 = Variable('i12', dtype=np.int32, initial=0, to_write=True)
                ts12 = Variable('ts12', dtype=np.int16, initial=0, to_write=True)
                dt12 = Variable('dt12', dtype=np.int16, initial=0, to_write=True)

                i13 = Variable('i13', dtype=np.int32, initial=0, to_write=True)
                ts13 = Variable('ts13', dtype=np.int16, initial=0, to_write=True)
                dt13 = Variable('dt13', dtype=np.int16, initial=0, to_write=True)

                i14 = Variable('i14', dtype=np.int32, initial=0, to_write=True)
                ts14 = Variable('ts14', dtype=np.int16, initial=0, to_write=True)
                dt14 = Variable('dt14', dtype=np.int16, initial=0, to_write=True)

                i15 = Variable('i15', dtype=np.int32, initial=0, to_write=True)
                ts15 = Variable('ts15', dtype=np.int16, initial=0, to_write=True)
                dt15 = Variable('dt15', dtype=np.int16, initial=0, to_write=True)

                i16 = Variable('i16', dtype=np.int32, initial=0, to_write=True)
                ts16 = Variable('ts16', dtype=np.int16, initial=0, to_write=True)
                dt16 = Variable('dt16', dtype=np.int16, initial=0, to_write=True)

                i17 = Variable('i17', dtype=np.int32, initial=0, to_write=True)
                ts17 = Variable('ts17', dtype=np.int16, initial=0, to_write=True)
                dt17 = Variable('dt17', dtype=np.int16, initial=0, to_write=True)

                i18 = Variable('i18', dtype=np.int32, initial=0, to_write=True)
                ts18 = Variable('ts18', dtype=np.int16, initial=0, to_write=True)
                dt18 = Variable('dt18', dtype=np.int16, initial=0, to_write=True)

                i19 = Variable('i19', dtype=np.int32, initial=0, to_write=True)
                ts19 = Variable('ts19', dtype=np.int16, initial=0, to_write=True)
                dt19 = Variable('dt19', dtype=np.int16, initial=0, to_write=True)

                ##################################################################
                # TEMPORARY TESTING VARIABLES ####################################
                ##################################################################

                # Number of larvae represented by particle
                N = Variable('N', dtype=np.float32, initial=1., to_write=True)

                # Larvae lost to sites
                Ns0 = Variable('Ns0', dtype=np.float32, initial=0., to_write=True)
                Ns1 = Variable('Ns1', dtype=np.float32, initial=0., to_write=True)
                Ns2 = Variable('Ns2', dtype=np.float32, initial=0., to_write=True)
                Ns3 = Variable('Ns3', dtype=np.float32, initial=0., to_write=True)
                Ns4 = Variable('Ns4', dtype=np.float32, initial=0., to_write=True)
                Ns5 = Variable('Ns5', dtype=np.float32, initial=0., to_write=True)
                Ns6 = Variable('Ns6', dtype=np.float32, initial=0., to_write=True)
                Ns7 = Variable('Ns7', dtype=np.float32, initial=0., to_write=True)
                Ns8 = Variable('Ns8', dtype=np.float32, initial=0., to_write=True)
                Ns9 = Variable('Ns9', dtype=np.float32, initial=0., to_write=True)
                Ns10 = Variable('Ns10', dtype=np.float32, initial=0., to_write=True)
                Ns11 = Variable('Ns11', dtype=np.float32, initial=0., to_write=True)
                Ns12 = Variable('Ns12', dtype=np.float32, initial=0., to_write=True)
                Ns13 = Variable('Ns13', dtype=np.float32, initial=0., to_write=True)
                Ns14 = Variable('Ns14', dtype=np.float32, initial=0., to_write=True)
                Ns15 = Variable('Ns15', dtype=np.float32, initial=0., to_write=True)
                Ns16 = Variable('Ns16', dtype=np.float32, initial=0., to_write=True)
                Ns17 = Variable('Ns17', dtype=np.float32, initial=0., to_write=True)
                Ns18 = Variable('Ns18', dtype=np.float32, initial=0., to_write=True)
                Ns19 = Variable('Ns19', dtype=np.float32, initial=0., to_write=True)

                # Number of larvae accumulated in the current reef
                Ns = Variable('Ns', dtype=np.float32, initial=0., to_write=True)
                N0 = Variable('N0', dtype=np.float32, initial=0., to_write=True)

                # Reef fraction
                rf = Variable('rf', dtype=np.float32, initial=0., to_write=True)
        else:
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
                i0 = Variable('i0', dtype=np.int32, initial=0, to_write=True)
                ts0 = Variable('ts0', dtype=np.int16, initial=0, to_write=True)
                dt0 = Variable('dt0', dtype=np.int16, initial=0, to_write=True)

                i1 = Variable('i1', dtype=np.int32, initial=0, to_write=True)
                ts1 = Variable('ts1', dtype=np.int16, initial=0, to_write=True)
                dt1 = Variable('dt1', dtype=np.int16, initial=0, to_write=True)

                i2 = Variable('i2', dtype=np.int32, initial=0, to_write=True)
                ts2 = Variable('ts2', dtype=np.int16, initial=0, to_write=True)
                dt2 = Variable('dt2', dtype=np.int16, initial=0, to_write=True)

                i3 = Variable('i3', dtype=np.int32, initial=0, to_write=True)
                ts3 = Variable('ts3', dtype=np.int16, initial=0, to_write=True)
                dt3 = Variable('dt3', dtype=np.int16, initial=0, to_write=True)

                i4 = Variable('i4', dtype=np.int32, initial=0, to_write=True)
                ts4 = Variable('ts4', dtype=np.int16, initial=0, to_write=True)
                dt4 = Variable('dt4', dtype=np.int16, initial=0, to_write=True)

                i5 = Variable('i5', dtype=np.int32, initial=0, to_write=True)
                ts5 = Variable('ts5', dtype=np.int16, initial=0, to_write=True)
                dt5 = Variable('dt5', dtype=np.int16, initial=0, to_write=True)

                i6 = Variable('i6', dtype=np.int32, initial=0, to_write=True)
                ts6 = Variable('ts6', dtype=np.int16, initial=0, to_write=True)
                dt6 = Variable('dt6', dtype=np.int16, initial=0, to_write=True)

                i7 = Variable('i7', dtype=np.int32, initial=0, to_write=True)
                ts7 = Variable('ts7', dtype=np.int16, initial=0, to_write=True)
                dt7 = Variable('dt7', dtype=np.int16, initial=0, to_write=True)

                i8 = Variable('i8', dtype=np.int32, initial=0, to_write=True)
                ts8 = Variable('ts8', dtype=np.int16, initial=0, to_write=True)
                dt8 = Variable('dt8', dtype=np.int16, initial=0, to_write=True)

                i9 = Variable('i9', dtype=np.int32, initial=0, to_write=True)
                ts9 = Variable('ts9', dtype=np.int16, initial=0, to_write=True)
                dt9 = Variable('dt9', dtype=np.int16, initial=0, to_write=True)

                i10 = Variable('i10', dtype=np.int32, initial=0, to_write=True)
                ts10 = Variable('ts10', dtype=np.int16, initial=0, to_write=True)
                dt10 = Variable('dt10', dtype=np.int16, initial=0, to_write=True)

                i11 = Variable('i11', dtype=np.int32, initial=0, to_write=True)
                ts11 = Variable('ts11', dtype=np.int16, initial=0, to_write=True)
                dt11 = Variable('dt11', dtype=np.int16, initial=0, to_write=True)

                i12 = Variable('i12', dtype=np.int32, initial=0, to_write=True)
                ts12 = Variable('ts12', dtype=np.int16, initial=0, to_write=True)
                dt12 = Variable('dt12', dtype=np.int16, initial=0, to_write=True)

                i13 = Variable('i13', dtype=np.int32, initial=0, to_write=True)
                ts13 = Variable('ts13', dtype=np.int16, initial=0, to_write=True)
                dt13 = Variable('dt13', dtype=np.int16, initial=0, to_write=True)

                i14 = Variable('i14', dtype=np.int32, initial=0, to_write=True)
                ts14 = Variable('ts14', dtype=np.int16, initial=0, to_write=True)
                dt14 = Variable('dt14', dtype=np.int16, initial=0, to_write=True)

                i15 = Variable('i15', dtype=np.int32, initial=0, to_write=True)
                ts15 = Variable('ts15', dtype=np.int16, initial=0, to_write=True)
                dt15 = Variable('dt15', dtype=np.int16, initial=0, to_write=True)

                i16 = Variable('i16', dtype=np.int32, initial=0, to_write=True)
                ts16 = Variable('ts16', dtype=np.int16, initial=0, to_write=True)
                dt16 = Variable('dt16', dtype=np.int16, initial=0, to_write=True)

                i17 = Variable('i17', dtype=np.int32, initial=0, to_write=True)
                ts17 = Variable('ts17', dtype=np.int16, initial=0, to_write=True)
                dt17 = Variable('dt17', dtype=np.int16, initial=0, to_write=True)

                i18 = Variable('i18', dtype=np.int32, initial=0, to_write=True)
                ts18 = Variable('ts18', dtype=np.int16, initial=0, to_write=True)
                dt18 = Variable('dt18', dtype=np.int16, initial=0, to_write=True)

                i19 = Variable('i19', dtype=np.int32, initial=0, to_write=True)
                ts19 = Variable('ts19', dtype=np.int16, initial=0, to_write=True)
                dt19 = Variable('dt19', dtype=np.int16, initial=0, to_write=True)

        return larva

    def build_event_kernel(self, test):
        """
        This script builds the event kernel as a test or operational kernel based
        on whether test is True or False

        """

        if type(test) != bool:
            raise Exception('Input must be a boolean.')

        if test:
            def event(particle, fieldset, time):

                # TESTING ONLY ############################################
                lm = 8e-7
                ls = 1e-5
                ###########################################################

                # 1 Keep track of the amount of time spent at sea
                particle.ot += 1

                # 2 Assess reef status
                particle.idx = fieldset.coral_idx_c[particle]

                # TESTING ONLY ############################################
                particle.rf = fieldset.coral_frac_c[particle]
                ###########################################################

                save_event = False
                new_event = False

                # 3 Trigger event cascade if larva is in a reef site and competency has been reached
                if particle.idx > 0 and particle.ot > fieldset.competency:

                    # Check if an event has already been triggered
                    if particle.current_reef_ts > 0:

                        # Check if we are in the same reef idx as the current event
                        if particle.idx == particle.current_reef_idx:

                            # If contiguous event, just add time and phi
                            particle.current_reef_ts += 1

                            # TESTING ONLY ############################################
                            particle.Ns = particle.Ns + (particle.rf*ls*particle.N*particle.dt)
                            particle.N  = particle.N - ((particle.rf*ls + lm)*particle.N*particle.dt)
                            particle.N0 = particle.rf*ls*particle.N*particle.dt
                            ###########################################################

                            # But also check that the particle isn't about to expire (save if so)
                            # Otherwise particles hanging around reefs at the end of the simulation
                            # won't get saved.

                            if particle.ot > fieldset.max_age:
                                save_event = True

                        else:

                            # TESTING ONLY ############################################
                            particle.Ns = particle.Ns
                            particle.N0 = particle.rf*ls*particle.N*particle.dt
                            particle.N  = particle.N - ((particle.rf*ls + lm)*particle.N*particle.dt)
                            ###########################################################

                            # Otherwise, we need to save the old event and create a new event
                            save_event = True
                            new_event = True

                    else:

                        # TESTING ONLY ############################################
                        particle.Ns = particle.Ns + (particle.rf*ls*particle.N*particle.dt)
                        particle.N0 = particle.rf*ls*particle.N*particle.dt
                        particle.N  = particle.N - ((particle.rf*ls + lm)*particle.N*particle.dt)
                        ###########################################################

                        # If event has not been triggered, create a new event
                        new_event = True

                else:

                    # Otherwise, check if ongoing event has just ended
                    if particle.current_reef_ts > 0:

                        # TESTING ONLY ############################################
                        particle.Ns = particle.Ns
                        particle.N0 = particle.rf*ls*particle.N*particle.dt
                        particle.N  = particle.N - ((particle.rf*ls + lm)*particle.N*particle.dt)
                        ###########################################################

                        save_event = True
                    else:
                        # TESTING ONLY ############################################
                        particle.N  = particle.N - (lm*particle.N*particle.dt)
                        ###########################################################


                if save_event:
                    # Save current values
                    # Unfortunately, due to the limited functions allowed in parcels, this
                    # required an horrendous if-else chain

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

                        particle.delete() # Delete particle, since no more reefs can be saved

                    # Then reset current values to zero
                    particle.current_reef_idx = 0
                    particle.current_reef_ts0 = 0
                    particle.current_reef_ts = 0
                    # particle.Ns = 0

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
                    particle.Ns = particle.N0
                    ###########################################################

                # Finally, check if particle needs to be deleted
                if particle.ot >= fieldset.max_age:

                    # Only delete particles where at least 1 event has been recorded
                    if particle.e_num > 0:
                        particle.delete()

        else:
            def event(particle, fieldset, time):

                # 1 Keep track of the amount of time spent at sea
                particle.ot += 1

                # 2 Assess reef status
                particle.idx = fieldset.coral_idx_c[particle]

                save_event = False
                new_event = False

                # 3 Trigger event cascade if larva is in a reef site and competency has been reached
                if particle.idx > 0 and particle.ot > fieldset.competency:

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

                        particle.delete() # Delete particle, since no more reefs can be saved

                    # Then reset current values to zero
                    particle.current_reef_idx = 0
                    particle.current_reef_ts0 = 0
                    particle.current_reef_ts = 0
                    # particle.Ns = 0

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
        Generate the ParticleSet object for OceanParcels

        Parameters
        ----------
        **kwargs : fh = file name for exported netcdf with trajectories

        """

        if not self.status['particleset']:
            raise Exception('Please run particleset first.')

        if 'fh' in kwargs.keys():
            self.fh['traj'] = self.dirs['traj'] + kwargs['fh']
        else:
            self.fh['traj'] = self.dirs['traj'] + 'example_output.nc'
            print('No output file handle provided - using defaults.' )

        print('Exporting output to ' + str(self.fh['traj']))

        if self.cfg['test']:
            if self.cfg['test_type'] == 'kernel':
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=24))
            else:
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=2))
        else:
            self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], write_ondelete=True)

        def deleteParticle(particle, fieldset, time):
            #  Recovery kernel to delete a particle if an error occurs
            particle.delete()

        # Run the simulation
        self.pset.execute(self.kernel,
                          runtime=self.cfg['run_time'],
                          dt=self.cfg['dt'],
                          recovery={ErrorCode.ErrorOutOfBounds: deleteParticle,
                                    ErrorCode.ErrorInterpolation: deleteParticle},
                          output_file=self.trajectory_file)

        # Export trajectory file
        self.trajectory_file.export()

        self.status['run'] = True



    def postrun_tests(self):
        """
        This function carries out a test for the accuracy of the events kernels
        or a close-up look of particle trajectories with respect to the model
        grid.
        (Note that this is inefficient legacy code, but it does work)

        The kernel function calculates the 'online' settling numbers from the
        test event kernel, and compares them to the analytical 'offline'
        numbers. If the offline analytical form works, the difference between
        the two should approach 0 as dt approaches 0.

        """

        if 'test' not in self.cfg.keys():
            raise Exception('Simulation must have been run in testing mode')
        elif not self.cfg['test']:
            raise Exception('Simulation must have been run in testing mode')
        elif not self.status['run']:
            raise Exception('Must run the simulation before events can be tested')

        if self.cfg['test_type'] == 'kernel':
            # Create a look-up table to relate cell index to coral reef fraction
            yidx_list, xidx_list = np.ma.nonzero(self.fields['idx'])
            idx_list = []
            fraction_list = []

            for (yidx, xidx) in zip(yidx_list, xidx_list):
                idx_list.append(self.fields['idx'][yidx, xidx])
                fraction_list.append(self.fields['cf'][yidx, xidx])

            cf_dict = dict(zip(idx_list, fraction_list))

            ls = self.cfg['test_parameters']['ls']
            lm = self.cfg['test_parameters']['lm']
            N0 = 1

            idx_arr = []
            ts_arr = []
            dt_arr = []

            Ns_arr = []
            Ns_pred_arr = []

            phi_arr = []
            cf_arr = []

            with Dataset(self.fh['traj'], mode='r') as nc:

                e_num = nc.variables['e_num'][:]
                e_num = e_num[np.ma.notmasked_edges(e_num, axis=1)[1]]

                pn = np.shape(e_num)[0]

                for i in range(19):
                    idx_arr_temp = nc.variables['i' + str(i)][:]
                    idx_arr.append(idx_arr_temp[np.ma.notmasked_edges(idx_arr_temp, axis=1)[1]])

                    ts_arr_temp = nc.variables['ts' + str(i)][:]
                    ts_arr.append(ts_arr_temp[np.ma.notmasked_edges(ts_arr_temp, axis=1)[1]])

                    dt_arr_temp = nc.variables['dt' + str(i)][:]
                    dt_arr.append(dt_arr_temp[np.ma.notmasked_edges(dt_arr_temp, axis=1)[1]])

                    Ns_arr_temp = nc.variables['Ns' + str(i)][:]
                    Ns_arr.append(Ns_arr_temp[np.ma.notmasked_edges(Ns_arr_temp, axis=1)[1]])

                    Ns_pred_temp = np.zeros_like(Ns_arr_temp[:, 0])
                    phi_temp = np.zeros_like(Ns_arr_temp[:, 0])
                    cf_temp = np.zeros_like(Ns_arr_temp[:, 0])

                    # For each particle, calculate the estimated loss per event using the equation
                    for j in range(pn):
                        if e_num[j] > i:
                            ij_idx = idx_arr[i][j]
                            ij_cf = cf_dict[ij_idx]

                            ij_ts0 = ts_arr[i][j]*self.cfg['dt'].total_seconds()
                            ij_dt = dt_arr[i][j]*self.cfg['dt'].total_seconds()

                            if i > 0:
                                ij_phi0 = phi_arr[i-1][j]
                            else:
                                ij_phi0 = 0

                            ij_coeff = N0*(ls*ij_cf)/(lm+(ls*ij_cf))
                            exp1 = np.exp((-ls*(ij_phi0+(ij_cf*ij_dt)))-(lm*(ij_ts0+ij_dt)))
                            exp0 = np.exp((-ls*ij_phi0)-(ij_ts0*lm))

                            Ns_pred_temp[j] = -ij_coeff*(exp1-exp0)

                            phi_temp[j] = ij_phi0 + ij_cf*ij_dt
                            cf_temp[j] = ij_cf

                    Ns_pred_arr.append(Ns_pred_temp)
                    phi_arr.append(phi_temp)
                    cf_arr.append(cf_temp)

            # Now calculate the errors
            Ns_all = np.array(Ns_arr).flatten() # Calculated (exact)
            Ns_pred_all = np.array(Ns_pred_arr).flatten() # Predicted

            # Calculate the difference
            pct_diff = 100*(Ns_pred_all - Ns_all)/(Ns_all)

            # Plot
            f, ax = plt.subplots(1, 1, figsize=(10, 10))

            xarg_max = np.max(np.abs(pct_diff[np.isfinite(pct_diff)]))
            ax.set_xlim([-xarg_max, xarg_max])
            ax.set_xlabel('Percentage difference between analytical and online settling fluxes')
            ax.set_ylabel('Number of events')
            ax.hist(pct_diff[np.isfinite(pct_diff)], range=(-xarg_max,xarg_max), bins=200, color='k')

            plt.savefig(self.dirs['fig'] + 'event_accuracy_test.png', dpi=300)
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

                plt.savefig(self.dirs['traj'] + 'trajectory_test.png', dpi=300)


































class Output():
    """
    Load and reformat output from coralsim

    """

    def __init__(self, root_dir):
        # Set up a status dictionary so we know the completion status of processes
        self.status = {'dict': False,
                       'dataframe': False}

    def generate_dict(self, fh, **kwargs):
        """
        Parameters (* are required)
        ----------
        kwargs :
            fh*: File handle for grid file
            idx_varname*: Variable name for the index grid
            cf_varname*: Variable name for the coral fraction grid
            cc_varname*: Variable name for the coral cover grid

        """

        if not all (key in kwargs.keys() for key in ['idx_varname', 'cf_varname', 'cc_varname', 'grp_varname']):
            raise KeyError('Must ensure that variable names for index, coral fraction and coral fraction grids have been supplied')

        with Dataset(fh, mode='r') as nc:
            self.coral_frac_grid = nc.variables[kwargs['cf_varname']][:]
            self.coral_cover_grid = nc.variables[kwargs['cc_varname']][:]
            self.coral_idx_grid = nc.variables[kwargs['idx_varname']][:]
            self.coral_grp_grid = nc.variables[kwargs['grp_varname']][:]

        # Find the locations of coral sites
        idx_list, cf_list, cc_list, grp_list = [], [], [], []

        for (yidx, xidx) in zip(np.ma.nonzero(self.coral_idx_grid)[0],
                                np.ma.nonzero(self.coral_idx_grid)[1]):

            # Translate index -> cf/cc
            idx_list.append(self.coral_idx_grid[yidx, xidx])
            cf_list.append(self.coral_frac_grid[yidx, xidx])
            cc_list.append(self.coral_cover_grid[yidx, xidx])

        self.cf_dict = dict(zip(idx_list, cf_list))
        self.cc_dict = dict(zip(idx_list, cc_list))
        self.cf_dict[0] = -999

        # Create dictionary to translate group -> number of cells in group
        grp_key, grp_val = np.unique(self.coral_grp_grid.compressed(),return_counts=True)
        self.num_in_grp_dict = dict(zip(grp_key, grp_val))

        self.status['dict'] = True


    def to_dataframe(self, fh_list, **kwargs):

        """
        Parameters (* are required)
        ----------
        kwargs :
            fh*: File handles to data
            lm*: Mortality rate for coral larvae
            ls*: Settling rate for coral larvae
            dt*: Model time-step (s)
            lpc*: Number of larvae released per cell
            rpm*: Number of releases per month

        """

        if not self.status['dict']:
            raise Exception('Please run generate_dict first')

        if not all (key in kwargs.keys() for key in ['lm', 'ls', 'dt', 'lpc', 'rpm']):
            raise KeyError('Please ensure that all parameters have been supplied')

        self.dt = kwargs['dt']
        self.lm = kwargs['lm']
        self.ls = kwargs['ls']

        dt = self.dt
        lm = self.lm
        ls = self.ls

        # Open the first file to find the number of events stored
        with Dataset(fh_list[0], mode='r') as nc:
            e_num = 0
            searching = True

            while searching:
                try:
                    nc.variables['i' + str(e_num)]
                    e_num += 1
                except:
                    searching = False

            self.max_events = e_num

        data_list = []

        # Now import all data
        for fhi, fh in tqdm(enumerate(fh_list), total=len(fh_list)):
            with Dataset(fh, mode='r') as nc:
                e_num = nc.variables['e_num'][:] # Number of events stored per trajectory
                n_traj = np.shape(e_num)[0] # Number of trajectories in file

                if not n_traj:
                    # Skip if there are no trajectories stored in file
                    continue

                # Extract origin date from filename
                y0 = int(fh.split('/')[-1].split('_')[1])
                m0 = int(fh.split('/')[-1].split('_')[2])
                d0 = int(fh.split('/')[-1].split('_')[-1].split('.')[0])
                t0 = datetime(year=y0, month=m0, day=d0, hour=0)

                # Load all data into memory
                idx_array = np.zeros((n_traj, self.max_events), dtype=np.uint16)
                ts0_array = np.zeros((n_traj, self.max_events), dtype=np.uint32)
                dts_array = np.zeros((n_traj, self.max_events), dtype=np.uint32)
                idx0_array = np.zeros((n_traj,), dtype=np.uint16)

                cf_array = np.zeros((n_traj, self.max_events), dtype=np.float32)
                ns_array = np.zeros((n_traj, self.max_events), dtype=np.float32)

                for i in range(self.max_events):
                    idx_array[:, i] = nc.variables['i' + str(i)][:, 0]
                    ts0_array[:, i] = nc.variables['ts' + str(i)][:, 0]*dt
                    dts_array[:, i] = nc.variables['dt' + str(i)][:, 0]*dt

                mask = (idx_array == 0)

                # Load remaining required variables
                idx0_array = nc.variables['idx0'][:]

            # Now generate an array containing the coral fraction for each index
            def translate(a, d):
                # Adapted from Maxim's excellent suggestion:
                # https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
                src, values = np.array(list(d.keys()), dtype=np.uint16), np.array(list(d.values()), dtype=np.float32)
                d_array = np.zeros((src.max()+1), dtype=np.float32)
                d_array[src] = values
                return d_array[a]

            cf_array = translate(idx_array, self.cf_dict)
            cf_array = np.ma.masked_array(cf_array, mask=mask)

            # Now calculate the fractional losses
            for i in range(self.max_events):
                if i == 0:
                    phi = np.zeros((n_traj,), dtype=np.float32)

                coeff = (ls*cf_array[:, i])/(lm+(ls*cf_array[:, i]))
                n0 = np.exp((-ls*phi)-(ts0_array[:, i]*lm))
                n1 = np.exp((-ls*(phi+(cf_array[:, i]*dts_array[:, i])))-(lm*(ts0_array[:, i]+dts_array[:, i])))

                ns_array[:, i] = -coeff*(n1-n0)
                phi = phi + cf_array[:, i]*dts_array[:, i]

            ns_array = np.ma.masked_array(ns_array, mask=mask)

            # From the index array, extract group
            grp_array = np.floor(idx_array/(2**8)).astype(np.uint8)
            grp_array = np.ma.masked_array(grp_array, mask=mask)

            # Extract origin group and project
            grp0 = np.floor(idx0_array/(2**8)).astype(np.uint8)
            grp0_array = np.zeros_like(idx_array, dtype=np.uint8)
            grp0_array[:] = grp0
            grp0_array = np.ma.masked_array(grp0_array, mask=mask)

            # Obtain origin coral cover as a proxy for total larval number and project
            cc0 = translate(idx0_array, self.cc_dict)
            cc0_array = np.zeros_like(idx_array, dtype=np.int32)
            cc0_array[:] = cc0
            cc0_array = np.ma.masked_array(cc0_array, mask=mask)
            cc0_array = cc0_array/kwargs['lpc']

            # Obtain number of larvae released in group for larval fraction and project
            lf0 = translate(grp0_array, self.num_in_grp_dict)
            lf0_array = np.zeros_like(idx_array, dtype=np.float32)
            lf0_array[:] = lf0
            lf0_array = 1/(kwargs['lpc']*kwargs['rpm']*lf0_array)

            # Convert fractional settling larvae to proportion of larvae released from source reef in release month
            settling_larvae_frac = lf0_array*ns_array

            # Convert fractional settling larvae to absolute number of settling, assuming num_released propto reef area
            settling_larvae = cc0_array*ns_array

            # Now insert into DataFrame
            frame = pd.DataFrame(data=grp0_array.compressed(), columns=['source_group'])
            frame['sink_group'] = grp_array.compressed()
            frame['settling_larval_frac'] = settling_larvae_frac.compressed()
            frame['settling_larval_num'] = settling_larvae.compressed()
            frame['release_year'] = y0
            frame['release_month'] = m0
            frame.reset_index(drop=True)

            frame.astype({'settling_larval_num': 'float32',
                          'release_year': 'int16',
                          'release_month': 'int8'})

            if kwargs['matrix']:
                grp_list = np.array([1, 3, 4, 5, 8, 11, 14, 15, 2, 6, 7, 9, 10, 12, 13, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 16, 22, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38])
                n_grp = len(grp_list)

                if fhi == 0:
                    probability_matrix = np.zeros((n_grp, n_grp), dtype=np.float32)
                    flux_matrix = np.zeros_like(probability_matrix, dtype=np.float32)

                for i, source_grp in enumerate(grp_list):
                    for j, sink_grp in enumerate(grp_list):
                        # Probability of a larva going from source -> sink:
                        # Sum of settling_larval_frac for source_group == source and
                        # sink_group == sink

                        # Flux of larvae going from source -> sink:
                        # Sum of settling larvae for source_group == source and
                        # sink_group == sink

                        subset = frame.loc[(frame['source_group'] == source_grp) &
                                           (frame['sink_group'] == sink_grp)].sum()

                        probability_matrix[i, j] += subset['settling_larval_frac']/(len(fh_list))
                        flux_matrix[i, j] += subset['settling_larval_num']/(len(fh_list)/(12*kwargs['rpm'])) # Convert to flux per year

        #     data_list.append(frame)

        # # Concatenate all frames
        # data = pd.concat(data_list, axis=0)
        # data.reset_index(drop=True)
        # self.data = data

        self.matrix = [probability_matrix, flux_matrix]

        self.status['dataframe'] = True

    def export_matrix(self, fh, **kwargs):
        """
        Parameters (* are required)
        ----------
        kwargs :
            fh*: Output file handle
            scheme*: Which region to generate a matrix for

        """

        if not self.status['dataframe']:
            raise Exception('Please run to_dataframe first')

        if 'scheme' not in kwargs:
            raise KeyError('Please specify a plotting scheme')
        elif kwargs['scheme'] not in ['seychelles']:
            raise KeyError('Scheme not understood')

        if kwargs['scheme'] == 'seychelles':
            # grp_list = self.data['source_group'].unique()
            # Reorder group list:
            # Farquhar Grp (8) | Aldabra Grp (7) | Alphonse Grp (2) | Amirantes (9) | Southern Coral Grp (2) | Inner Islands (10)
            grp_list = np.array([1, 3, 4, 5, 8, 11, 14, 15, 2, 6, 7, 9, 10, 12, 13, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 16, 22, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38])
            n_grp = len(grp_list)

            probability_matrix = self.matrix[0]
            flux_matrix = self.matrix[1]

            # for i, source_grp in enumerate(grp_list):
            #     for j, sink_grp in enumerate(grp_list):
            #         # Probability of a larva going from source -> sink:
            #         # Sum of settling_larval_frac for source_group == source and
            #         # sink_group == sink

            #         # Flux of larvae going from source -> sink:
            #         # Sum of settling larvae for source_group == source and
            #         # sink_group == sink

            #         subset = self.data.loc[(self.data['source_group'] == source_grp) &
            #                                (self.data['sink_group'] == sink_grp)].sum()

            #         probability_matrix[i, j] = subset['settling_larval_frac']/(12*kwargs['n_years'])
            #         flux_matrix[i, j] = subset['settling_larval_num']/(kwargs['n_years']) # Convert to flux per year

            f, ax = plt.subplots(1, 1, figsize=(13, 10), constrained_layout=True)
            font = {'family': 'normal',
                    'weight': 'normal',
                    'size': 16,}
            matplotlib.rc('font', **font)

            # Set up plot
            axis = np.linspace(0, n_grp, num=n_grp+1)
            i_coord, j_coord = np.meshgrid(axis, axis)
            pmatrix = ax.pcolormesh(i_coord, j_coord, probability_matrix, cmap=cmr.gem,
                                    norm=colors.LogNorm(vmin=1e-10, vmax=1e-2), shading='auto')

            # Adjust plot
            cax1 = f.add_axes([ax.get_position().x1+0.0,ax.get_position().y0-0.115,0.020,ax.get_position().height+0.231])

            cb1 = plt.colorbar(pmatrix, cax=cax1, pad=0.1)
            cb1.set_label('Probabiity of connection', size=16)

            ax.set_aspect('equal', adjustable=None)
            ax.margins(x=-0.01, y=-0.01)
            ax.xaxis.set_ticks(np.arange(39))
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticks(np.arange(39))
            ax.yaxis.set_ticklabels([])
            ax.set_xlim([0, 38])
            ax.set_ylim([0, 38])

            ax.tick_params(color='w', labelcolor='w')
            for spine in ax.spines.values():
                spine.set_edgecolor('w')

            # Add dividers
            for pos in [8, 15, 17, 26, 28, 38]:
                ax.plot(np.array([pos, pos]), np.array([0, 38]), '-', color='w', linewidth=2)
                ax.plot(np.array([0, 38]), np.array([pos, pos]), '-', color='w', linewidth=2)

            for spine in cax1.spines.values():
                spine.set_edgecolor('w')

            plt.savefig(fh, dpi=300, bbox_inches='tight')


            # # Add labels
            # for pos, label in zip([4, 11.5, 16, 21.5, 27, 33],
            #                       ['Farquhar Grp', 'Aldabra Grp', 'Alphonse Grp',
            #                        'Amirantes', 'Southern Coral Grp', 'Inner Islands']:
            #     ax.text()


            # plt.xticks([], "", ax=ax)
            # ax.set_ticks(np.arange(39))
            # ax.set_xlabel("")






# f, ax = plt.subplots(1, 1, figsize=(24, 10), constrained_layout=True,
#                      subplot_kw={'projection': ccrs.PlateCarree()})

# data_crs = ccrs.PlateCarree()
# coral = ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(coral_grid_w == 0, coral_grid_w)[1:-1, 1:-1],
#                        norm=colors.LogNorm(vmin=1e2, vmax=1e8), cmap=cmr.flamingo_r, transform=data_crs)
# ax.pcolormesh(lon_psi_w, lat_psi_w, np.ma.masked_where(lsm_rho_w == 0, 1-lsm_rho_w)[1:-1, 1:-1],
#               vmin=-2, vmax=1, cmap=cmr.neutral, transform=data_crs)

# gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', linestyle='-')
# gl.xlocator = mticker.FixedLocator(np.arange(35, 95, 5))
# gl.ylocator = mticker.FixedLocator(np.arange(-25, 5, 5))
# gl.ylabels_right = False
# gl.xlabels_top = False

# ax.set_xlim([34.62, 77.5])
# ax.set_ylim([-23.5, 0])
# ax.spines['geo'].set_linewidth(1)
# ax.set_ylabel('Latitude')
# ax.set_xlabel('Longitude')
# ax.set_title('Coral cells on WINDS grid (postproc)')

# cax1 = f.add_axes([ax.get_position().x1+0.07,ax.get_position().y0-0.10,0.015,ax.get_position().height+0.196])

# cb1 = plt.colorbar(oceanc, cax=cax1, pad=0.1)
# cb1.set_label('Coral surface area in cell (m2)', size=12)

# ax.set_aspect('equal', adjustable=None)
# ax.margins(x=-0.01, y=-0.01)

# plt.savefig(fh['fig'] + '_WINDS_postproc.png', dpi=300)








