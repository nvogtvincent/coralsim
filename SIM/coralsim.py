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
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import warnings
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar, Variable,
                     DiffusionUniformKh, ParcelsRandom)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime



class experiment():
    """
    Initialise a larval dispersal experiment.
    -----------
    Functions:
        import_grid
        import_currents
        set_release_time:
        set_release_loc:
        set_release_params:
        set_partitions:
        set_
    """

    def __init__(self, root_dir):
        # Set up a status dictionary so we know the completion status of the
        # experiment configuration

        self.status = {'grid': False,
                       'release_time': False,
                       'release_loc': False,
                       'release_params': False,
                       'partitions': False}

        # Set up dictionaries for various parameters
        self.fh = {}
        self.params = {}

        self.root_dir = root_dir

    def import_grid(self, **kwargs):
        """

        Parameters (* are required)
        ----------
        kwargs :
            fh* : File handle to grid netcdf file (if not already passed)
            grid_type* : 'C' or 'A' (C-grid or A-grid')
            dimensions* : Dict containing dimension names for the grid in the
                          following format:

                              A-GRID:
                              {'lon': LON_NAME,
                               'lat': LAT_NAME}

                              C-GRID:
                              {'rho': {'lon': LON_RHO_NAME,
                                       'lat': LAT_RHO_NAME},
                               'psi': {'lon': LON_PSI_NAME,
                                       'lat': LAT_PSI_NAME}}

        """

        if not all (key in kwargs.keys() for key in ['grid_type', 'dimensions']):
            raise KeyError('Please make sure all required arguments are passed')
        elif 'grid' not in self.fh.keys() and 'fh' in kwargs.keys():
            self.fh['grid'] = kwargs['fh']
        elif 'grid' not in self.fh.keys():
            raise KeyError('Please make sure all required arguments are passed')

        # Check that grid types and dimensions are supplied correctly
        if kwargs['grid_type'] in ['C', 'A']:
            self.params['grid_type'] = kwargs['grid_type']
            if self.params['grid_type'] == 'C':
                # Import C-Grid dimension names
                raise NotImplementedError('C-grids have not been implemented yet!')
            else:
                # Import A-Grid dimension names
                if all (key in kwargs['dimensions'].keys() for key in ['lon', 'lat']):
                    self.params['dimension_names'] = kwargs['dimensions']
                else:
                    raise Exception(('Please supply the \'lon\' and \'lat\' '
                                     'dimension names as a dictionary'))
        else:
            raise Exception('Grid type must be \'C\' or \'A\'')

        # Import coordinates for grids
        if self.params['grid_type'] == 'A':
            with Dataset(self.fh['grid'], mode='r') as nc:
                self.axes = {'rho': {},
                             'psi': {}}
                self.axes['rho']['lon'] = np.array(nc.variables[self.params['dimension_names']['lon']][:])
                self.axes['rho']['lat'] = np.array(nc.variables[self.params['dimension_names']['lat']][:])
                self.axes['rho']['nx'] = len(self.axes['rho']['lon'])
                self.axes['rho']['ny'] = len(self.axes['rho']['lat'])

                # Now generate the psi grid
                self.axes['psi']['lon'] = 0.5*(self.axes['rho']['lon'][1:] + self.axes['rho']['lon'][:-1])
                self.axes['psi']['lat'] = 0.5*(self.axes['rho']['lat'][1:] + self.axes['rho']['lat'][:-1])
                self.axes['psi']['nx'] = len(self.axes['psi']['lon'])
                self.axes['psi']['ny'] = len(self.axes['psi']['lat'])

        self.status['grid'] = True

    def import_currents(self, **kwargs):
        """

        Parameters (* are required)
        ----------
        kwargs :
            fh* : File handle/s to current data
            variables* : Dictionary of variable names (in OceanParcels format)
            dimensions* : Dictionary of dimension names (in OceanParcels format)
            interp_method: Parcels interpolation method, otherwise defaults for
                           A-Grid ('linear') and C-Grid ('cgrid_velocity')

        """

        if not all (key in kwargs.keys() for key in ['fh', 'variables', 'dimensions']):
            raise KeyError('Please make sure all required arguments are passed')
        else:
            self.fh['currents'] = kwargs['fh']

        if 'interp_method' in kwargs:
            interp_method = kwargs['interp_method']
        elif self.status['grid']:
            if self.params['grid_type'] == 'A':
                interp_method = 'linear'
            else:
                interp_method = 'cgrid_velocity'
        else:
            raise Exception('Either specify an interp method, or run import_grid first.')

        # Create the velocity fieldset
        self.fieldset = FieldSet.from_netcdf(filenames=kwargs['fh'],
                                             variables=kwargs['variables'],
                                             dimensions={'U': kwargs['dimensions'],
                                                         'V': kwargs['dimensions']},
                                             interp_method={'U': interp_method,
                                                            'V': interp_method})

        self.status['currents'] = True

    def create_particles(self, **kwargs):
        """

        Parameters (* are required)
        ----------
        kwargs :
            fh : File handle to grid netcdf file (if not already present)
            num_per_cell : Number of particles to (aim to) initialise per cell
            eez_list: List of EEZs to release particles from
            eez_varname: Variable name of EEZ grid in grid file
            grp: List of groups to release particles from
            grp_var_name: Variable name of group grid in grid file

        """

        if not self.status['grid']:
            raise Exception('Must import grid before particles can be generated!')
        elif 'num_per_cell' not in kwargs.keys():
            raise KeyError('Please make sure \'num_per_cell\' has been passed as an argument')
        elif 'coral_varname' not in kwargs.keys():
            raise KeyError('Must pass coral_varname as an argument')

        if 'export_grp' not in kwargs.keys():
            kwargs['export_grp'] = False

        if 'export_eez' not in kwargs.keys():
            kwargs['export_eez'] = False

        if 'export_coral_cover' not in kwargs.keys():
            kwargs['export_coral_cover'] = False

        if 'eez_filter' not in kwargs.keys():
            kwargs['eez_filter'] = False

        if 'grp_filter' not in kwargs.keys():
            kwargs['grp_filter'] = False

        # Import EEZ/Group grids if necessary
        if kwargs['eez_filter'] or kwargs['export_eez']:
            if 'eez_varname' not in kwargs.keys():
                raise KeyError('Please give the EEZ variable name if you want to filter by EEZ')
            with Dataset(self.fh['grid'], mode='r') as nc:
                eez_grid = nc.variables[kwargs['eez_varname']][:]

            if self.params['grid_type'] == 'A':
                # Make sure dimensions agree
                assert np.array_equiv(np.shape(eez_grid), (self.axes['psi']['ny'],
                                                           self.axes['psi']['nx']))
            else:
                raise NotImplementedError('C-grids have not been implemented yet!')

        if kwargs['grp_filter'] or kwargs['export_grp']:
            if 'grp_varname' not in kwargs.keys():
                raise KeyError('Please give the group variable name if you want to filter by group')
            with Dataset(self.fh['grid'], mode='r') as nc:
                grp_grid = nc.variables[kwargs['grp_varname']][:]

            if self.params['grid_type'] == 'A':
                # Make sure dimensions agree
                assert np.array_equiv(np.shape(grp_grid), (self.axes['psi']['ny'],
                                                           self.axes['psi']['nx']))
            else:
                raise NotImplementedError('C-grids have not been implemented yet!')

        # Import coral mask
        with Dataset(self.fh['grid'], mode='r') as nc:
            coral_grid = nc.variables[kwargs['coral_varname']][:]

        if self.params['grid_type'] == 'A':
            # Make sure dimensions agree
            assert np.array_equiv(np.shape(coral_grid), (self.axes['psi']['ny'],
                                                         self.axes['psi']['nx']))
        else:
            raise NotImplementedError('C-grids have not been implemented yet!')

        # Build a mask of valid locations
        coral_mask = (coral_grid > 0)
        if kwargs['eez_filter']:
            coral_mask *= np.isin(eez_grid, kwargs['eez_filter'])
        if kwargs['grp_filter']:
            coral_mask *= np.isin(grp_grid, kwargs['grp_filter'])

        nl = int(np.sum(coral_mask)) # Number of sites identified

        if nl == 0:
            raise Exception('No valid coral sites found')
        else:
            print(str(nl) + ' coral sites identified.')
            print()

        coral_yidx, coral_xidx = np.where(coral_mask)

        # Calculate the number of particles to release per cell (if not square)
        self.params['pn'] = int(np.ceil(kwargs['num_per_cell']**0.5))
        self.params['pn2'] = int(self.params['pn']**2)

        if np.ceil(kwargs['num_per_cell']**0.5) != kwargs['num_per_cell']**0.5:
            print('Particle number per cell is not square.')
            print('Old particle number: ' + str(kwargs['num_per_cell']))
            print('New particle number: ' + str(int(self.params['pn']**2)))
            print()

        # Build a list of initial particle locations
        lon_rho_grid, lat_rho_grid = np.meshgrid(self.axes['rho']['lon'],
                                                 self.axes['rho']['lat'])
        lon_psi_grid, lat_psi_grid = np.meshgrid(self.axes['psi']['lon'],
                                                 self.axes['psi']['lat'])

        self.particles = {} # Dictionary to hold initial particle properties
        self.particles['lon'] = np.zeros((nl*self.params['pn2'],), dtype=np.float64)
        self.particles['lat'] = np.zeros((nl*self.params['pn2'],), dtype=np.float64)
        if kwargs['export_coral_cover']:
            if np.max(coral_grid) < np.iinfo(np.int32).max:
                self.particles['coral_cover'] = np.zeros((nl*self.params['pn2'],),
                                                         dtype=np.int32)
            else:
                self.particles['coral_cover'] = np.zeros((nl*self.params['pn2'],),
                                                         dtype=np.int64)
        if kwargs['export_eez']:
            if np.max(eez_grid) < np.iinfo(np.int8).max:
                self.particles['eez'] = np.zeros((nl*self.params['pn2'],),
                                                 dtype=np.int8)
            else:
                self.particles['eez'] = np.zeros((nl*self.params['pn2'],),
                                                 dtype=np.int16)

        if kwargs['export_grp']:
            if np.max(grp_grid) < np.iinfo(np.int8).max:
                self.particles['grp'] = np.zeros((nl*self.params['pn2'],),
                                                 dtype=np.int8)
            else:
                self.particles['grp'] = np.zeros((nl*self.params['pn2'],),
                                                 dtype=np.int16)

        if self.params['grid_type'] == 'A':
            # For cell psi[i, j], the surrounding rho cells are:
            # rho[i, j]     (SW)
            # rho[i, j+1]   (SE)
            # rho[i+1, j]   (NW)
            # rho[i+1, j+1] (NE)

            for k, (i, j) in enumerate(zip(coral_yidx, coral_xidx)):
                # Firstly calculate the basic particle grid (may be variable for
                # curvilinear grids)

                dX = lon_rho_grid[i, j+1] - lon_rho_grid[i, j] # Grid spacing
                dY = lat_rho_grid[i+1, j] - lat_rho_grid[i, j] # Grid spacing
                dx = dX/self.params['pn']                      # Particle spacing
                dy = dY/self.params['pn']                      # Particle spacing

                gx = np.linspace(lon_rho_grid[i, j]+(dx/2),    # Particle x locations
                                 lon_rho_grid[i, j+1]-(dx/2), num=self.params['pn'])

                gy = np.linspace(lat_rho_grid[i, j]+(dy/2),    # Particle y locations
                                 lat_rho_grid[i+1, j]-(dy/2), num=self.params['pn'])

                gx, gy = [grid.flatten() for grid in np.meshgrid(gx, gy)] # Flattened arrays

                self.particles['lon'][k*self.params['pn2']:(k+1)*self.params['pn2']] = gx
                self.particles['lat'][k*self.params['pn2']:(k+1)*self.params['pn2']] = gy

                if kwargs['export_coral_cover']:
                    coral_cover_k = coral_grid[i, j]
                    self.particles['coral_cover'][k*self.params['pn2']:(k+1)*self.params['pn2']] = coral_cover_k

                if kwargs['export_eez']:
                    eez_k = eez_grid[i, j]
                    self.particles['eez'][k*self.params['pn2']:(k+1)*self.params['pn2']] = eez_k

                if kwargs['export_grp']:
                    grp_k = grp_grid[i, j]
                    self.particles['grp'][k*self.params['pn2']:(k+1)*self.params['pn2']] = grp_k

        else:
            raise NotImplementedError('C-grids have not been implemented yet!')

        kwargs['plot'] = False if 'plot' not in kwargs.keys() else kwargs['plot']

        if kwargs['plot']:
            # Create a plot of initial particle positions
            kwargs['plot_colour'] = None if 'plot_colour' not in kwargs.keys() else kwargs['plot_colour']
            if kwargs['plot_colour'] == 'grp':
                if kwargs['export_grp']:
                    colour_series = self.particles['grp']
                else:
                    raise Exception('Cannot plot by group if group is not exported ')
            elif kwargs['plot_colour'] == 'eez':
                if kwargs['export_eez']:
                    colour_series = self.particles['eez']
                else:
                    raise Exception('Cannot plot by group if EEZ is not exported ')
            else:
                colour_series = 'k'

            plot_x_range = np.max(self.particles['lon']) - np.min(self.particles['lon'])
            plot_y_range = np.max(self.particles['lat']) - np.min(self.particles['lat'])
            plot_x_range = [np.min(self.particles['lon']) - 0.1*plot_x_range,
                            np.max(self.particles['lon']) + 0.1*plot_x_range]
            plot_y_range = [np.min(self.particles['lat']) - 0.1*plot_y_range,
                            np.max(self.particles['lat']) + 0.1*plot_y_range]
            aspect = (plot_y_range[1] - plot_y_range[0])/(plot_x_range[1] - plot_x_range[0])

            f, ax = plt.subplots(1, 1, figsize=(20, 20*aspect), subplot_kw={'projection': ccrs.PlateCarree()})
            cmap = 'prism'

            ax.set_xlim(plot_x_range)
            ax.set_ylim(plot_y_range)
            ax.set_title('Initial positions for particles')

            ax.scatter(self.particles['lon'], self.particles['lat'], c=colour_series,
                       cmap=cmap, s=1, transform=ccrs.PlateCarree())

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.8, color='black', linestyle='-', zorder=11)
            gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 5))
            gl.xlocator = mticker.FixedLocator(np.arange(0, 90, 5))
            gl.xlabels_top = False
            gl.ylabels_right = False

            if 'plot_fh' in kwargs.keys():
                plot_fh = kwargs['plot_fh']
            else:
                warnings.warn('No plotting fh given, saving to script directory')
                plot_fh = self.root_dir + 'initial_particle_positions.png'

            plt.savefig(plot_fh, dpi=300)
            plt.close()















